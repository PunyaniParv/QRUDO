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

Diagnostic / verification tools
-------------------------------
    python -m voice.wake_word                  # live listener: per-second RMS/peak,
                                               # raw model score, running max, threshold
    python -m voice.wake_word --test-wav f.wav # score an existing WAV file
    python -m voice.wake_word --record-test    # record ~8 s from the selected mic,
                                               # save a WAV, then score it
    python -m voice.wake_word --speak-on-detect
        # With any of the above: on detection, print and speak the demo reply
        # "Hey, how are you?" via offline TTS (Windows System.Speech). Runs the
        # full microsecond-scale chain: mic -> wake word -> spoken response.

``--test-wav`` reports the raw (unfiltered) model score through the exact
same predict path the live listener uses, so it separates a microphone/audio
problem from a model problem. ``--record-test`` records with the same
microphone-selection logic as the listener, then runs the same WAV scoring.
Neither command needs the internet.

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
import tempfile
import time
import wave
import numpy as np

from voice.config import CONFIG
from voice.detect import WakeDetector
from voice.device import MicrophoneStream
from voice.log import voice_debug, voice_trace
from voice import tts

logger = logging.getLogger("sarv.voice.wake_word")

# openWakeWord expects 16 kHz int16 mono audio in 80 ms frames.
SAMPLE_RATE = 16000
FRAME_SIZE = 1280  # 80 ms at 16 kHz

# Spoken back when the wake word is detected under --speak-on-detect.
_RESPONSE = "Hey, how are you?"

def _rms_int16(chunk: np.ndarray) -> float:
    """Calculate RMS level of an int16 audio chunk.

    Args:
        chunk: 1D numpy array of int16 samples.

    Returns:
        RMS value in the int16 scale (roughly 0-32768).
    """
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def _peak_abs_int16(chunk: np.ndarray) -> float:
    """Calculate peak absolute amplitude of an int16 audio chunk.

    Args:
        chunk: 1D numpy array of int16 samples.

    Returns:
        Peak absolute value in the int16 scale (0-32768).
    """
    return float(np.max(np.abs(chunk.astype(np.int16))))


# ---------------------------------------------------------------------------
# WAV-based verification tools (offline, no internet access)
# ---------------------------------------------------------------------------
def _wav_temp_dir() -> str:
    """A per-user temp folder (outside the repo) for recorded test WAVs."""
    return os.path.join(tempfile.gettempdir(), "qrudo_voice")


def _read_wav(path: str) -> tuple:
    """Read a PCM WAV file.

    Returns ``(samples_2d, channels, sample_rate, n_frames)`` where
    ``samples_2d`` is an int16 array of shape ``(n_frames, channels)``.
    Raises :class:`ValueError` for non-16-bit or unreadable files.
    """
    try:
        with wave.open(path, "rb") as f:
            params = f.getparams()
            sample_rate = params.framerate
            channels = params.nchannels
            sampwidth = params.sampwidth
            n_frames = params.nframes
            if sampwidth != 2:
                raise ValueError(
                    f"WAV must be 16-bit PCM (this file is {sampwidth * 8}-bit): {path}"
                )
            raw = f.readframes(n_frames)
    except (OSError, wave.Error) as exc:
        raise ValueError(f"could not read WAV {path!r}: {exc}") from exc
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
    return samples, channels, sample_rate, n_frames


def _to_mono_16k(samples_2d: np.ndarray, sample_rate: int) -> tuple:
    """Convert WAV samples to mono 16 kHz int16 1-D audio.

    Downmixes multi-channel audio by averaging, resamples to 16 kHz with
    numpy linear interpolation (no extra dependencies) when needed, and
    clips to the int16 range. Returns ``(samples_1d, resampled)`` where
    ``resampled`` tells the caller whether the sample rate was changed.
    """
    if samples_2d.ndim == 1:
        mono = samples_2d.astype(np.float64)
    else:
        mono = np.mean(samples_2d, axis=1).astype(np.float64)
    if len(mono) == 0:
        return np.array([], dtype=np.int16), False
    if sample_rate == SAMPLE_RATE:
        return np.round(mono).astype(np.int16), False
    n_out = int(round(len(mono) * SAMPLE_RATE / sample_rate))
    x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    resampled = np.interp(x_new, x_old, mono)
    resampled = np.clip(np.round(resampled), -32768.0, 32767.0)
    return resampled.astype(np.int16), True


def _score_wav(engine, samples: np.ndarray, pad_seconds: float = 1.0) -> tuple:
    """Score mono 16 kHz int16 ``samples`` with the engine's model.

    Prepends ``pad_seconds`` of silence so openWakeWord's ~5-frame warmup and
    its streaming mel-spectrogram get real context -- the same convention
    ``openwakeword.Model.predict_clip`` uses. Frames are fed to ``predict`` one
    at a time, exactly like the live listener, with NO patience/threshold
    filtering, so the true raw model score is visible.

    Returns ``(max_scores, best_frames)``: every prediction key mapped to its
    peak raw score and to the clip frame index (0-based, after padding) where
    that peak occurred.
    """
    reset = getattr(engine._model, "reset", None)
    if callable(reset):
        reset()
    pad = np.zeros(int(pad_seconds * SAMPLE_RATE), dtype=np.int16)
    data = np.concatenate([pad, samples])
    pad_frames = pad.shape[0] // FRAME_SIZE
    max_scores: dict[str, float] = {}
    best_frames: dict[str, int] = {}
    frame = 0
    for start in range(0, data.shape[0], FRAME_SIZE):
        chunk = data[start:start + FRAME_SIZE]
        if chunk.shape[0] != FRAME_SIZE:
            chunk = LocalWakeWordEngine._pad_or_trim(chunk)
        scores = engine._model.predict(chunk)  # raw scores (no patience/threshold)
        clip_frame = frame - pad_frames
        if clip_frame >= 0:
            for key, value in scores.items():
                score = float(value)
                if score > max_scores.get(key, 0.0):
                    max_scores[key] = score
                    best_frames[key] = clip_frame
        frame += 1
    return max_scores, best_frames


def _respond_if_wanted(speak_on_detect: bool) -> None:
    """Print and speak the demo reply, but only when ``--speak-on-detect`` is set.

    Called exactly once per detection (callers gate on the rising edge), so a
    single utterance never produces repeated responses. Speech happens after
    the ``[qrudo]`` line is printed, matching the demo contract.
    """
    if not speak_on_detect:
        return
    print(f"[qrudo] {_RESPONSE}")
    try:
        tts.speak(_RESPONSE)
    except tts.TTSError as exc:
        print(f"[qrudo] ERROR: could not speak the response: {exc}")


def _run_wav_test(engine, path: str, speak_on_detect: bool = False) -> int:
    """Score a WAV file against the loaded model and print the result."""
    print("[wake-word] WAV test")
    print(f"[wake-word] File: {path}")
    try:
        samples, channels, sample_rate, n_frames = _read_wav(path)
    except ValueError as exc:
        print(f"[wake-word] ERROR: {exc}")
        return 1
    print(f"[wake-word] Sample rate: {sample_rate} Hz")
    print(f"[wake-word] Channels: {channels}")
    print(f"[wake-word] Frames: {n_frames}")

    mono, resampled = _to_mono_16k(samples, sample_rate)
    if resampled:
        print(
            f"[wake-word] Note: resampled {sample_rate} Hz -> {SAMPLE_RATE} Hz "
            "for the model (linear interpolation)"
        )

    max_scores, best_frames = _score_wav(engine, mono)
    model_key = max(max_scores, key=max_scores.get) if max_scores else None
    max_score = max_scores.get(model_key, 0.0) if model_key else 0.0
    best_frame = best_frames.get(model_key, -1)

    print(f"[wake-word] Max score: {max_score:.3f}")
    if best_frame >= 0:
        best_time = best_frame * FRAME_SIZE / SAMPLE_RATE
        print(f"[wake-word] Peak score at: {best_time:.2f} s (frame {best_frame})")
    print(f"[wake-word] Detection threshold: {engine._threshold:.2f}")

    detected = model_key is not None and max_score >= engine._threshold
    print(f"[wake-word] RESULT: {'DETECTED' if detected else 'NOT DETECTED'}")
    if detected:
        _respond_if_wanted(speak_on_detect)
    return 0


def _record_wav(path: str, seconds: float) -> None:
    """Record ``seconds`` of mono 16 kHz int16 audio from the selected mic."""
    n_chunks = max(1, int(seconds * SAMPLE_RATE / FRAME_SIZE))
    frames: list[np.ndarray] = []
    with MicrophoneStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SIZE,
    ) as stream:
        for _ in range(n_chunks):
            frame, _overflowed = stream.read(FRAME_SIZE)
            frames.append(np.asarray(frame, dtype=np.int16).reshape(-1))
    audio = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(audio.tobytes())


def _run_record_test(engine, args) -> int:
    """Record from the selected mic to a WAV, then score it with the model."""
    seconds = max(1.0, float(getattr(args, "seconds", 8.0)))
    path = getattr(args, "record_test", None) or ""
    if not path:
        out_dir = _wav_temp_dir()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir,
            f"qrudo_wake_test_{time.strftime('%Y%m%d_%H%M%S')}.wav",
        )
    spec, mic_name = _report_microphone()
    print(f"[wake-word] Recording {seconds:.0f} s from: {mic_name} (spec={spec})")
    print("[wake-word] Say the wake phrase now...")
    try:
        _record_wav(path, seconds)
    except Exception as exc:
        print(f"[wake-word] ERROR: recording failed: {exc}")
        return 1
    print(f"[wake-word] Saved: {path}")
    return _run_wav_test(engine, path, speak_on_detect=getattr(args, "speak_on_detect", False))


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

    def reset(self) -> None:
        """Clear the detector's internal state (default: no-op).

        Called by the command pipeline right after a detection so the
        wake-word state settles -- stale detection/audio buffers cannot
        re-trigger during the next listening session. Engines with streaming
        state override this (see LocalWakeWordEngine); engines without any
        are unaffected.
        """

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

    def wait_for_wake_word(
        self,
        frame_source=None,
        stop=None,
        debug: bool = False,
    ) -> bool:
        """Block until the wake word is heard; return True on detection.

        The detection criterion lives in :class:`~voice.detect.WakeDetector`
        (peak trigger + sliding-window patience + cooldown) -- measured on
        real audio, the old 3-consecutive-frames rule missed genuine phrases
        whose raw peak lasted a single 80 ms frame.

        Two ways to be fed audio:

        * ``frame_source`` (callable) -- the caller owns the microphone and
          hands frames over one at a time (signature
          ``frame_source(timeout, stop)``, returning an int16 1-D frame or
          None on timeout/stop).  The voice pipeline passes its shared
          ``MicMonitor.next_frame`` so wake detection and command capture see
          the same uninterrupted stream.
        * ``frame_source=None`` -- the engine opens its own microphone stream
          (standalone use: ``run_voice_loop``, the CLI diagnostics).

        Returns False when ``stop()`` turns true or shutdown ends the frame
        source, so callers can distinguish "heard nothing, stopping" from
        "wake word heard".  ``debug=True`` prints per-second ``[wake-debug]``
        lines and a ``[wake-stats]`` block on detection.

        Why raw predict instead of openWakeWord's model-side ``patience``?
        openWakeWord 0.6.0's ``predict(..., patience=..., threshold=...)``
        filter zeroes a frame unless the previous 3 *filtered* predictions in
        its internal buffer were all above the threshold. That buffer is
        seeded with zeros by the model's 5-frame warmup and by the filter's
        own zeroing, so the condition can never become true -- it returns 0.0
        forever on audio that raw ``predict()`` scores above 0.5. Patience is
        therefore applied here in Python, over raw scores.
        """
        if self._model is None:
            self.initialize()

        cfg = self.config
        detector = WakeDetector(
            threshold=self._threshold,
            window=int(getattr(cfg, "wake_window", 4)),
            window_min=int(getattr(cfg, "wake_window_min", 2)),
            peak_threshold=float(getattr(cfg, "wake_peak_threshold", 0.65)),
            peak_support=float(getattr(cfg, "wake_peak_support", 0.3)),
            support_window=int(getattr(cfg, "wake_support_window", 6)),
            cooldown_s=float(getattr(cfg, "wake_cooldown_s", 1.0)),
            debug=debug,
        )

        last_report = time.monotonic()

        def _score(samples: np.ndarray) -> bool:
            nonlocal last_report
            scores = self._model.predict(samples)
            name = self._model_names[0] if self._model_names else "?"
            value = float(scores.get(name, 0.0))
            now = time.monotonic()
            if now - last_report >= 1.0:
                voice_trace(
                    "[wake-debug] frame={} RMS={:.1f} raw_score={:.3f} "
                    "threshold={:.2f} max_score={:.3f}".format(
                        detector.frames,
                        _rms_int16(samples),
                        value,
                        self._threshold,
                        detector.max_score,
                    ),
                    enabled=debug,
                )
                last_report = now
            if detector.update(value):
                if debug:
                    voice_debug("[wake-stats]", enabled=debug)
                    for key, val in detector.stats().items():
                        voice_debug(f"{key}={val}", enabled=debug)
                    voice_debug("[wake-debug] WAKE DETECTED", enabled=debug)
                return True
            return False

        if frame_source is not None:
            empty_polls = 0
            while True:
                if stop is not None and stop():
                    return False
                frame = frame_source(timeout=0.25, stop=stop)
                if frame is None:
                    if stop is not None and stop():
                        return False
                    empty_polls += 1
                    if empty_polls >= 3:
                        raise WakeWordError("microphone stream ended")
                    continue
                empty_polls = 0
                samples = np.asarray(frame, dtype=np.int16).reshape(-1)
                if samples.shape[0] != self.frame_size:
                    samples = self._pad_or_trim(samples)
                if _score(samples):
                    return True

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
                if _score(samples):
                    return True
        finally:
            stream.__exit__(None, None, None)

    @staticmethod
    def _pad_or_trim(samples: np.ndarray) -> np.ndarray:
        if samples.shape[0] == FRAME_SIZE:
            return samples
        if samples.shape[0] > FRAME_SIZE:
            return samples[:FRAME_SIZE]
        return np.pad(samples, (0, FRAME_SIZE - samples.shape[0]))

    def reset(self) -> None:
        """Clear openWakeWord's prediction/audio buffers (see interface docs).

        ``openwakeword.Model.reset()`` wipes the streaming prediction and
        mel-spectrogram buffers so the *next* ``wait_for_wake_word()`` session
        starts clean instead of inheriting the just-detected wake phrase's
        stale high scores.
        """
        reset = getattr(self._model, "reset", None)
        if callable(reset):
            reset()

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
    parser.add_argument("--test-wav", metavar="PATH",
                        help="Score a 16-bit PCM WAV file against the wake-word "
                             "model and report the max score (no internet needed).")
    parser.add_argument("--record-test", nargs="?", const="", default=None,
                        metavar="PATH",
                        help="Record from the selected mic to a WAV (default ~8 s), "
                             "save it, then run --test-wav on it. An optional PATH "
                             "chooses where the WAV is written.")
    parser.add_argument("--seconds", type=float, default=8.0,
                        help="Recording length for --record-test (default 8).")
    parser.add_argument("--speak-on-detect", action="store_true",
                        help="When the wake word is detected, speak the demo "
                             "reply ('Hey, how are you?') through the system "
                             "audio output using offline TTS.")
    args = parser.parse_args(argv)

    if args.list_models:
        return _list_models()
    if args.download:
        return _download_models()

    if args.record_test is not None or args.test_wav:
        # Offline, deterministic WAV verification: --test-wav needs no mic, and
        # --record-test uses the exact same scoring path afterwards.
        try:
            engine = create_wake_word_engine(CONFIG)
        except WakeWordError as exc:
            print(f"[wake-word] ERROR: {exc}")
            return 1
        try:
            engine.initialize()
            print(f"[wake-word] Model loaded   : {engine.model_path}")
        except WakeWordError as exc:
            print(f"[wake-word] ERROR: {exc}")
            return 1
        try:
            if args.record_test is not None:
                return _run_record_test(engine, args)
            return _run_wav_test(
                engine, args.test_wav, speak_on_detect=args.speak_on_detect
            )
        finally:
            engine.close()

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

    # One detection state machine per loaded model (the same criterion
    # wait_for_wake_word uses), so the CLI reports what the engine decides.
    cfg = CONFIG
    detectors = {
        name: WakeDetector(
            threshold=engine._threshold,
            window=int(getattr(cfg, "wake_window", 4)),
            window_min=int(getattr(cfg, "wake_window_min", 2)),
            peak_threshold=float(getattr(cfg, "wake_peak_threshold", 0.65)),
            peak_support=float(getattr(cfg, "wake_peak_support", 0.3)),
            support_window=int(getattr(cfg, "wake_support_window", 6)),
            cooldown_s=float(getattr(cfg, "wake_cooldown_s", 1.0)),
        )
        for name in engine._model_names
    }

    # Diagnostic state.
    max_seen = {name: 0.0 for name in engine._model_names}
    last_report_time = time.monotonic()
    report_interval = 1.0  # seconds
    frame_counter = 0
    fired_names = set()

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
                # --- DIAGNOSTIC CALCULATIONS (no alteration of audio data) ---
                samples = np.asarray(frame, dtype=np.int16).reshape(-1)
                if samples.shape[0] != engine.frame_size:
                    samples = LocalWakeWordEngine._pad_or_trim(samples)
                rms = _rms_int16(samples)
                peak = _peak_abs_int16(samples)

                # ONE model call per frame, with no patience/threshold filtering,
                # so the raw model score is visible. The same raw scores feed
                # the WakeDetector that wait_for_wake_word uses.
                scores = engine._model.predict(samples)
                for model_name, score in scores.items():
                    max_seen[model_name] = max(
                        max_seen.get(model_name, 0.0), float(score)
                    )

                now = time.monotonic()

                # Periodic diagnostic reports (every ~1 s): RMS, peak, current
                # model score, max score seen, and the detection threshold.
                if now - last_report_time >= report_interval:
                    # Distinguish audio state:
                    if rms < 100:
                        audio_state = "SILENCE"
                    elif rms < 1000:
                        audio_state = "normal speech"
                    else:
                        audio_state = "loud speech"
                    print(
                        f"[diag] frame {frame_counter} | RMS={rms:5.1f} peak={peak:5.1f} | audio={audio_state}"
                    )
                    for model_name, score in scores.items():
                        # Distinguish model score state:
                        if score < 0.3:
                            score_state = "low"
                        elif score < engine._threshold:
                            score_state = "below threshold"
                        else:
                            score_state = "high / detection"
                        print(
                            f"[diag] score {model_name}={float(score):.3f} ({score_state}) "
                            f"| max={max_seen[model_name]:.3f} "
                            f"| threshold={engine._threshold:.2f}"
                        )
                    last_report_time = now

                for model_name, score in scores.items():
                    hit = detectors[model_name].update(float(score))
                    if hit and model_name not in fired_names:
                        fired_names.add(model_name)
                        print(f"[wake-word] WAKE WORD DETECTED: {model_name}")
                        print("[wake-word] Ready for command.")
                        _respond_if_wanted(args.speak_on_detect)

                frame_counter += 1
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

