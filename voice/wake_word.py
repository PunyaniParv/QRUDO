"""
Local, replaceable wake-word layer for QRUDO.

The rest of the voice pipeline talks only to the :class:`WakeWordEngine`
interface and to :func:`create_wake_word_engine`. Nothing outside this module
imports openWakeWord (or Porcupine) directly, so the underlying engine can be
swapped later without touching ``voice/pipeline.py``.

Engine backends
---------------
* ``openwakeword`` (default) -- :class:`LocalWakeWordEngine`.
  Runs fully offline, no API key, no account, no per-user cost, and supports
  both Windows and macOS (via the onnxruntime backend). The engine *code* is
  Apache-2.0.

  License caveat: openWakeWord's bundled *pretrained* models are licensed
  CC BY-NC-SA 4.0 (non-commercial). They are fine for local development and
  manual testing, but they MUST NOT be redistributed in a commercial QRUDO
  build. For a commercial product, train a custom "Hey QRUDO" model on
  permissive data and point ``wake_word_model_path`` at it.

Model provisioning
------------------
The engine is deliberately *offline and deterministic*: it loads whatever
model file exists on disk and never downloads anything inside the detect loop.
Provision the bundled models once for development/testing with:

    python -m voice.wake_word --download

Then run the diagnostic listener with:

    python -m voice.wake_word

Training a custom "Hey QRUDO" model
-----------------------------------
openWakeWord supports training a custom wake phrase (its code is Apache-2.0).
Because the bundled pretrained models are non-commercial, a commercial QRUDO
build MUST ship its own model trained on permissive data. High-level process:

1. Record a large set of positive examples of "Hey QRUDO" (many speakers,
   devices, room acoustics) and a large set of negative examples (background
   noise, music, other speech).
2. Follow the official openWakeWord training guide (see the repo's
   ``docs``/``notebooks`` for training a new model in ~1 hour via Colab).
   Verify the training data licensing allows commercial use.
3. Export the trained model to ONNX.
4. Set ``SARV_WAKE_WORD_MODEL_PATH`` to that file (or place it and set
   ``wake_word_model_path`` in voice/config.py). The engine then loads it
   instead of the bundled pretrained model.

This is the only remaining step before "Hey QRUDO" works in a commercial build.
Until then, the engine can be exercised for manual testing with the bundled
models (development/testing only).
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np

from voice.config import CONFIG
from voice.device import MicrophoneStream

logger = logging.getLogger("sarv.voice.wake_word")

# openWakeWord expects 16 kHz int16 mono audio in 80 ms frames.
SAMPLE_RATE = 16000
FRAME_SIZE = 1280  # 80 ms at 16 kHz


class WakeWordError(Exception):
    """Base class for anticipated wake-word failures (not programming bugs)."""


class WakeWordModelNotFoundError(WakeWordError):
    """Raised when the configured wake-word model file cannot be found."""


class WakeWordEngineError(WakeWordError):
    """Raised when a wake-word engine fails to initialize or run."""


class WakeWordEngine:
    """Stable interface every wake-word backend implements.

    QRUDO's voice pipeline depends only on this interface, so swapping the
    underlying engine (openWakeWord now, a Porcupine or custom-trained model
    later) never touches ``voice/pipeline.py``.

    A typical lifecycle is::

        engine = create_wake_word_engine()
        engine.initialize()          # load the model once, reuse it
        engine.wait_for_wake_word()  # block until the phrase is heard
        engine.close()               # release resources (idempotent)
    """

    name = "base"
    sample_rate = SAMPLE_RATE
    frame_size = FRAME_SIZE

    def initialize(self) -> None:
        """Load the wake-word model and inference resources exactly once.

        Never called per audio frame. Raises :class:`WakeWordError` (or a
        subclass) on anticipated failures such as a missing model file.
        """
        raise NotImplementedError

    def wait_for_wake_word(self) -> None:
        """Block until the wake word/phrase is heard, then return.

        Uses :class:`~voice.device.MicrophoneStream`, so it always hears from
        the same, already-configured microphone (built-in, USB, Bluetooth or
        OS default) and inherits the existing device-selection/fallback logic.
        """
        raise NotImplementedError
    def close(self) -> None:
        """Release microphone/model/inference resources. Must be idempotent."""
        raise NotImplementedError



class LocalWakeWordEngine(WakeWordEngine):
    """openWakeWord-based local wake-word engine.

    The model is loaded once in :meth:`initialize` and reused across every
    :meth:`wait_for_wake_word` call. Audio is read in 80 ms frames
    (1280 samples @ 16 kHz) -- the chunk size openWakeWord is designed for.
    """

    name = "openwakeword"

    def __init__(self, config=None) -> None:
        self.config = config if config is not None else CONFIG
        self.model_path = ""
        self._model = None
        self._model_names: list[str] = []
        self._threshold = float(getattr(self.config, "wake_word_sensitivity", 0.5))

    # -- model path resolution --------------------------------------------
    def resolve_model_path(self):
        """Return the absolute model file to load (or ``None`` if unknown).

        Precedence:
          1. ``wake_word_model_path`` -- a custom model (e.g. a trained
             "Hey QRUDO" model). If set but missing, a clear error is raised.
          2. ``wake_word_model_name`` -- a bundled openWakeWord pretrained
             model. If its file has not been downloaded yet, returns the
             would-be path so the caller can report it.
        """
        cfg = self.config
        custom = getattr(cfg, "wake_word_model_path", None)
        if custom:
            if not os.path.isfile(custom):
                raise WakeWordModelNotFoundError(
                    f"QRUDO wake-word model not found: {custom}"
                )
            return os.path.abspath(custom)

        try:
            import openwakeword
        except Exception as exc:  # pragma: no cover - env-dependent
            raise WakeWordEngineError(
                "openwakeword is not installed. Add it via requirements-voice.txt "
                "and try again."
            ) from exc

        base_dir = os.path.join(
            os.path.dirname(openwakeword.__file__), "resources", "models"
        )
        known: dict[str, str] = {}
        for key, info in openwakeword.MODELS.items():
            fname = os.path.basename(info["download_url"]).replace(".tflite", ".onnx")
            known[key] = os.path.join(base_dir, fname)

        name = str(getattr(cfg, "wake_word_model_name", "hey_jarvis")).lower()
        name = name.replace(" ", "_")
        if name in known:
            return known[name]
        hits = [path for key, path in known.items() if name in key]
        return hits[0] if hits else None

    def initialize(self) -> None:
        """Load the openWakeWord model. Safe to call multiple times."""
        if self._model is not None:
            return
        path = self.resolve_model_path()
        if path is None:
            raise WakeWordModelNotFoundError(
                "QRUDO wake-word model not found: no model is configured. "
                "Set SARV_WAKE_WORD_MODEL_PATH to a model file, or run "
                "`python -m voice.wake_word --download` to fetch a bundled model."
            )
        if not os.path.isfile(path):
            raise WakeWordModelNotFoundError(
                f"QRUDO wake-word model not found: {path}. Run "
                "`python -m voice.wake_word --download` to fetch a bundled "
                "model, or point SARV_WAKE_WORD_MODEL_PATH at a model file."
            )
        self.model_path = path
        self._model = self._build_model(path)
        self._model_names = list(self._model.models.keys())

    def _build_model(self, path):
        try:
            import openwakeword
        except Exception as exc:  # pragma: no cover - env-dependent
            raise WakeWordEngineError(
                "openwakeword is not installed. Add it via requirements-voice.txt."
            ) from exc
        try:
            return openwakeword.Model(wakeword_models=[path], inference_framework="onnx")
        except FileNotFoundError as exc:
            raise WakeWordModelNotFoundError(
                f"QRUDO wake-word model resources not found ({exc}). Run "
                "`python -m voice.wake_word --download` to fetch the "
                "openWakeWord feature models."
            ) from exc

    def wait_for_wake_word(self) -> None:
        """Block until a wake-word frame scores above the threshold."""
        if self._model is None:
            self.initialize()

        patience = {name: 3 for name in self._model_names}
        threshold = {name: self._threshold for name in self._model_names}

        # Only the stream-open is wrapped so real detection errors surface.
        try:
            stream = MicrophoneStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.frame_size,
            )
            stream.__enter__()
        except Exception as exc:
            raise WakeWordError(f"No usable microphone found: {exc}") from exc

        try:
            while True:
                frame, _overflowed = stream.read(self.frame_size)
                samples = np.asarray(frame, dtype=np.int16).reshape(-1)
                if samples.shape[0] != self.frame_size:
                    samples = self._pad_or_trim(samples)
                scores = self._model.predict(
                    samples,
                    patience=patience,
                    threshold=threshold,
                    debounce_time=0.5,
                )
                if any(score >= self._threshold for score in scores.values()):
                    return
        finally:
            stream.__exit__(None, None, None)

    @staticmethod
    def _pad_or_trim(samples: np.ndarray) -> np.ndarray:
        if samples.shape[0] == FRAME_SIZE:
            return samples
        if samples.shape[0] > FRAME_SIZE:
            return samples[:FRAME_SIZE]
        return np.pad(samples, (0, FRAME_SIZE - samples.shape[0]))

    def close(self) -> None:
        """Release model/inference resources (idempotent)."""
        self._model = None
        self._model_names = []
        self.model_path = ""


def create_wake_word_engine(config=None) -> WakeWordEngine:
    """Factory: return a :class:`WakeWordEngine` for ``config.wake_word_engine``.

    This is the only place that knows which concrete engine is being used, so
    adding a new engine (e.g. Porcupine or a custom-trained model) is a one-line
    change here -- nothing else in the pipeline needs to know.
    """
    cfg = config if config is not None else CONFIG
    engine_name = str(getattr(cfg, "wake_word_engine", "openwakeword")).strip().lower()
    if engine_name in ("openwakeword", "local", "open_wake_word"):
        return LocalWakeWordEngine(cfg)
    raise WakeWordEngineError(
        f"Unknown wake_word_engine {engine_name!r}. Supported: 'openwakeword'."
    )

# ---------------------------------------------------------------------------
# Manual / diagnostic entry point:  python -m voice.wake_word
# ---------------------------------------------------------------------------
def _report_microphone() -> tuple:
    # Imported here so importing the module never touches hardware.
    from voice.device import _resolve

    return _resolve(CONFIG.microphone_device)


def _download_models() -> int:
    try:
        import openwakeword
        from openwakeword.utils import download_models
    except Exception as exc:
        print(f"[wake-word] ERROR: openwakeword not installed ({exc})")
        return 1
    name = getattr(CONFIG, "wake_word_model_name", "hey_jarvis")
    print(f"[wake-word] Downloading openWakeWord feature models + {name!r} ...")
    try:
        download_models(model_names=[name])
    except Exception as exc:
        print(f"[wake-word] ERROR: download failed: {exc}")
        return 1
    print("[wake-word] Download complete.")
    print("[wake-word] NOTE: bundled models are CC BY-NC-SA 4.0 (non-commercial); "
          "development/testing only. Ship a custom-trained model in a commercial build.")
    return 0


def _list_models() -> int:
    try:
        import openwakeword
    except Exception as exc:
        print(f"[wake-word] ERROR: openwakeword not installed ({exc})")
        return 1
    print("[wake-word] Bundled openWakeWord model names:")
    for key in openwakeword.MODELS:
        print(f"  - {key}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voice.wake_word",
        description="QRUDO wake-word diagnostic.",
    )
    parser.add_argument("--download", action="store_true",
                        help="Download bundled openWakeWord models, then exit.")
    parser.add_argument("--list-models", action="store_true",
                        help="List known bundled openWakeWord model names.")
    args = parser.parse_args(argv)

    if args.list_models:
        return _list_models()
    if args.download:
        return _download_models()

    try:
        engine = create_wake_word_engine(CONFIG)
    except WakeWordError as exc:
        print(f"[wake-word] ERROR: {exc}")
        return 1

    spec, mic_name = _report_microphone()
    print(f"[wake-word] Engine         : {engine.name}")
    print(f"[wake-word] Microphone     : {mic_name} (spec={spec})")
    print(f"[wake-word] Sample rate    : {engine.sample_rate} Hz")
    print(f"[wake-word] Frame size     : {engine.frame_size} samples "
          f"({engine.frame_size / engine.sample_rate * 1000:.0f} ms)")

    try:
        engine.initialize()
        print(f"[wake-word] Model loaded   : {engine.model_path}")
        print(f"[wake-word] Detection thres: {engine._threshold:.2f}")
        print("[wake-word] Listening... say the wake phrase. Ctrl+C to stop.")
    except WakeWordError as exc:
        print(f"[wake-word] ERROR: {exc}")
        return 1

    patience = {name: 3 for name in engine._model_names}
    threshold = {name: engine._threshold for name in engine._model_names}
    try:
        try:
            stream = MicrophoneStream(
                samplerate=engine.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=engine.frame_size,
            )
            stream.__enter__()
        except Exception as exc:
            print(f"[wake-word] ERROR: No usable microphone found: {exc}")
            return 1
        try:
            while True:
                frame, _overflowed = stream.read(engine.frame_size)
                samples = np.asarray(frame, dtype=np.int16).reshape(-1)
                scores = engine._model.predict(
                    samples, patience=patience, threshold=threshold, debounce_time=0.5
                )
                for model_name, score in scores.items():
                    if score >= engine._threshold:
                        print(f"[wake-word] DETECTED  {model_name}: {score:.3f}")
        finally:
            stream.__exit__(None, None, None)
    except KeyboardInterrupt:
        print("\n[wake-word] Stopped by user.")
        return 0
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

