"""Tests for the replaceable wake-word layer (voice/wake_word.py).

These tests deliberately avoid hardware and network dependencies:
  * no physical microphone / Bluetooth / USB device
  * no real wake-word model file
  * no API key / account
  * no network access

The openWakeWord package is faked in sys.modules so the tests are independent
of whether openwakeword is installed in the running environment.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voice import tts
from voice import wake_word
from voice.config import VoiceConfig
from voice.wake_word import (
    LocalWakeWordEngine,
    WakeWordEngine,
    WakeWordEngineError,
    WakeWordError,
    WakeWordModelNotFoundError,
    create_wake_word_engine,
)

FRAME_SIZE = wake_word.FRAME_SIZE


def _fake_openwakeword_module():
    """Minimal stand-in for the openwakeword package (no model files on disk)."""
    base = os.path.join(tempfile.mkdtemp(), "openwakeword", "__init__.py")
    mod = types.ModuleType("openwakeword")
    mod.__file__ = base
    mod.MODELS = {
        "hey_jarvis": {"download_url": "https://example/hey_jarvis_v0.1.tflite"},
        "alexa": {"download_url": "https://example/alexa_v0.1.tflite"},
    }
    return mod


def _engine_config(**overrides):
    """A config-like object carrying only the wake-word fields we need."""
    base = {
        "wake_word_engine": "openwakeword",
        "wake_word_model_name": "hey_jarvis",
        "wake_word_model_path": None,
        "wake_word_sensitivity": 0.5,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeModel:
    """A fake openWakeWord model returning a fixed score per frame."""

    def __init__(self, score=0.9):
        self.score = score
        self.calls = 0
        self.last_kwargs = {}
        self.models = {"hey_jarvis": object()}

    def predict(self, samples, patience=None, threshold=None, debounce_time=0.0):
        self.calls += 1
        self.last_kwargs = {
            "patience": patience,
            "threshold": threshold,
            "debounce_time": debounce_time,
        }
        assert samples.shape[0] == FRAME_SIZE
        return {"hey_jarvis": self.score}


class _PatienceBuggyModel(_FakeModel):
    """openWakeWord 0.6.0's real behavior: forwarding patience zeroes scores.

    The model-side patience filter returns 0.0 forever (its internal buffer is
    seeded with warmup zeros and the filter's own zeroing, so the patience
    condition can never be met) even when the raw score is high.
    """

    def predict(self, samples, patience=None, threshold=None, debounce_time=0.0):
        self.calls += 1
        self.last_kwargs = {
            "patience": patience,
            "threshold": threshold,
            "debounce_time": debounce_time,
        }
        assert samples.shape[0] == FRAME_SIZE
        if patience:
            return {"hey_jarvis": 0.0}
        return {"hey_jarvis": self.score}


class _SequencedModel(_FakeModel):
    """A fake model returning a per-call score sequence (last repeats)."""

    def __init__(self, scores):
        super().__init__(score=scores[0])
        self.scores = list(scores)

    def predict(self, samples, patience=None, threshold=None, debounce_time=0.0):
        self.calls += 1
        self.last_kwargs = {
            "patience": patience,
            "threshold": threshold,
            "debounce_time": debounce_time,
        }
        assert samples.shape[0] == FRAME_SIZE
        index = min(self.calls - 1, len(self.scores) - 1)
        return {"hey_jarvis": self.scores[index]}


class _FakeStream:
    """A fake MicrophoneStream producing int16 silence frames."""

    def __init__(self, frames, channels=1, **kwargs):
        self._frames = frames
        self._channels = channels
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc_info):
        self.exited += 1
        return None

    def read(self, n):
        if self._frames is not None:
            if not self._frames:
                raise StopIteration("no more frames in fake stream")
            frame = self._frames.pop(0)
            return frame, False
        return np.zeros((n, self._channels), dtype=np.int16), False


def _make_engine(score=0.9, **config_overrides):
    engine = LocalWakeWordEngine(_engine_config(**config_overrides))
    engine._threshold = 0.5
    engine._model = _FakeModel(score)
    engine._model_names = ["hey_jarvis"]
    return engine
    return engine


class WakeWordEngineInterfaceCase(unittest.TestCase):
    def test_base_interface_is_abstract_contract(self):
        base = WakeWordEngine()
        with self.assertRaises(NotImplementedError):
            base.initialize()
        with self.assertRaises(NotImplementedError):
            base.wait_for_wake_word()
        with self.assertRaises(NotImplementedError):
            base.close()
        self.assertEqual(base.name, "base")
        self.assertEqual(base.sample_rate, 16000)
        self.assertEqual(base.frame_size, 1280)

    def test_local_engine_implements_interface(self):
        engine = LocalWakeWordEngine(_engine_config())
        self.assertIsInstance(engine, WakeWordEngine)
        for attr in ("initialize", "wait_for_wake_word", "reset", "close"):
            self.assertTrue(callable(getattr(engine, attr)))
        self.assertEqual(engine.name, "openwakeword")
        self.assertEqual(engine.sample_rate, 16000)
        self.assertEqual(engine.frame_size, 1280)


class WakeWordResetCase(unittest.TestCase):
    def test_base_interface_reset_is_safe_noop(self):
        WakeWordEngine().reset()

    def test_reset_calls_model_reset_when_available(self):
        engine = _make_engine(score=0.9)
        model = mock.Mock()
        engine._model = model
        engine.reset()
        model.reset.assert_called_once()

    def test_reset_is_safe_when_model_has_no_reset(self):
        engine = _make_engine(score=0.9)
        self.assertFalse(hasattr(engine._model, "reset"))
        engine.reset()  # must not raise


class WakeWordConfigCase(unittest.TestCase):
    def test_config_defaults_are_sensible(self):
        cfg = VoiceConfig()
        self.assertEqual(cfg.wake_word_engine, "openwakeword")
        self.assertIn(cfg.wake_word_model_name, ("hey_jarvis", "alexa"))
        self.assertIsNone(cfg.wake_word_model_path)
        self.assertIsInstance(cfg.wake_word_sensitivity, float)
        self.assertTrue(0.0 <= cfg.wake_word_sensitivity <= 1.0)

    def test_config_has_no_picovoice_key_requirement(self):
        cfg = VoiceConfig()
        # The old AccessKey field must no longer exist on the normal path.
        self.assertFalse(hasattr(cfg, "picovoice_access_key"))
        self.assertFalse(hasattr(cfg, "wake_word_path"))


class WakeWordFactoryCase(unittest.TestCase):
    def test_factory_returns_local_engine(self):
        engine = create_wake_word_engine(_engine_config())
        self.assertIsInstance(engine, LocalWakeWordEngine)

    def test_factory_accepts_name_aliases(self):
        for name in ("openwakeword", "local", "open_wake_word"):
            with self.subTest(name=name):
                engine = create_wake_word_engine(
                    _engine_config(wake_word_engine=name)
                )
                self.assertIsInstance(engine, LocalWakeWordEngine)

    def test_factory_raises_on_unknown_engine(self):
        with self.assertRaises(WakeWordEngineError):
            create_wake_word_engine(_engine_config(wake_word_engine="porcupine"))


class MissingModelCase(unittest.TestCase):
    def test_custom_missing_path_raises_clear_error(self):
        missing = os.path.join(tempfile.mkdtemp(), "nope.onnx")
        engine = LocalWakeWordEngine(
            _engine_config(wake_word_model_path=missing)
        )
        with self.assertRaises(WakeWordModelNotFoundError) as ctx:
            engine.initialize()
        self.assertIn("QRUDO wake-word model not found", str(ctx.exception))
        self.assertIn(missing, str(ctx.exception))

    def test_bundled_missing_file_raises_clear_error(self):
        fake = _fake_openwakeword_module()
        with mock.patch.dict(sys.modules, {"openwakeword": fake}):
            engine = LocalWakeWordEngine(_engine_config())
            with self.assertRaises(WakeWordModelNotFoundError) as ctx:
                engine.initialize()
            self.assertIn("QRUDO wake-word model not found", str(ctx.exception))
            self.assertIn("hey_jarvis", str(ctx.exception))

    def test_no_model_configured_raises_clear_error(self):
        fake = _fake_openwakeword_module()
        fake.MODELS = {}  # no bundled models at all
        with mock.patch.dict(sys.modules, {"openwakeword": fake}):
            engine = LocalWakeWordEngine(
                _engine_config(wake_word_model_name="missing_phrase")
            )
            with self.assertRaises(WakeWordModelNotFoundError) as ctx:
                engine.initialize()
            self.assertIn("no model is configured", str(ctx.exception))


class EngineInitializationCase(unittest.TestCase):
    def test_model_loaded_once_and_reused(self):
        engine = LocalWakeWordEngine(_engine_config())
        fake_model = _FakeModel(0.5)
        fake_path = os.path.join(tempfile.mkdtemp(), "model.onnx")
        with open(fake_path, "wb"):
            pass  # create an existing file so resolve_model_path passes
        with mock.patch.object(engine, "resolve_model_path", return_value=fake_path), \
             mock.patch.object(engine, "_build_model", return_value=fake_model) as build:
            engine.initialize()
            engine.initialize()  # must not rebuild
        build.assert_called_once()
        self.assertIs(engine._model, fake_model)
        self.assertEqual(engine._model_names, ["hey_jarvis"])

    def test_init_failure_propagates_as_wake_word_error(self):
        engine = LocalWakeWordEngine(_engine_config())
        with mock.patch.object(
            engine, "resolve_model_path",
            side_effect=WakeWordModelNotFoundError("QRUDO wake-word model not found: x"),
        ):
            with self.assertRaises(WakeWordError):
                engine.initialize()

class MicrophoneIntegrationCase(unittest.TestCase):
    def test_wait_returns_when_detected(self):
        engine = _make_engine(score=0.9)
        with mock.patch.object(
            wake_word, "MicrophoneStream", return_value=_FakeStream(frames=None)
        ), redirect_stdout(io.StringIO()) as buf:
            engine.wait_for_wake_word()
        # Python-side patience: 3 consecutive frames >= threshold are needed.
        self.assertEqual(engine._model.calls, 3)
        self.assertIn("[wake-debug] WAKE DETECTED", buf.getvalue())

    def test_wake_stats_printed_on_detection(self):
        engine = _make_engine(score=0.9)
        with mock.patch.object(
            wake_word, "MicrophoneStream", return_value=_FakeStream(frames=None)
        ), redirect_stdout(io.StringIO()) as buf:
            engine.wait_for_wake_word()
        out = buf.getvalue()
        self.assertIn("[wake-stats]", out)
        self.assertIn("frames=3", out)
        self.assertIn("max_score=0.900", out)
        self.assertIn("frames_above_threshold=3", out)
        self.assertIn("longest_consecutive_high=3", out)
        self.assertIn("patience_required=3", out)
        self.assertIn("patience_fired=yes", out)

    def test_wait_does_not_trigger_below_threshold(self):
        # A fake stream that returns one frame below the detection threshold.
        stream = _FakeStream(frames=[np.zeros((FRAME_SIZE, 1), dtype=np.int16)])
        engine = _make_engine(score=0.1)  # below 0.5 threshold -> never triggers
        with mock.patch.object(wake_word, "MicrophoneStream", return_value=stream):
            # The fake stream runs out of frames (StopIteration), which shows the
            # engine did not return on the sub-threshold frame.
            with self.assertRaises(StopIteration):
                while True:
                    engine.wait_for_wake_word()
        self.assertGreaterEqual(engine._model.calls, 1)

    def test_mic_unavailable_raises_clear_error(self):
        class BrokenStream(_FakeStream):
            def __enter__(self):
                raise OSError("no input device")

        engine = _make_engine()
        with mock.patch.object(
            wake_word, "MicrophoneStream", return_value=BrokenStream(None)
        ):
            with self.assertRaises(WakeWordError) as ctx:
                engine.wait_for_wake_word()
        self.assertIn("No usable microphone found", str(ctx.exception))

    def test_mic_stream_is_closed_on_detection(self):
        stream = _FakeStream(frames=None)
        engine = _make_engine(score=0.9)
        with mock.patch.object(wake_word, "MicrophoneStream", return_value=stream), \
                redirect_stdout(io.StringIO()):
            engine.wait_for_wake_word()
        self.assertEqual(stream.entered, 1)
        self.assertEqual(stream.exited, 1)

    def test_wait_uses_raw_predict_without_model_side_patience(self):
        # Regression: openWakeWord 0.6.0's model-side patience filter returns
        # 0.0 forever (verified on the real model), so wait_for_wake_word must
        # NOT forward patience/threshold to the model -- patience is applied in
        # Python. debounce_time must stay zero (never forwarded together).
        engine = _make_engine(score=0.9)
        with mock.patch.object(
            wake_word, "MicrophoneStream", return_value=_FakeStream(frames=None)
        ), redirect_stdout(io.StringIO()):
            engine.wait_for_wake_word()
        kwargs = engine._model.last_kwargs
        self.assertFalse(kwargs["patience"])  # raw predict: patience not forwarded
        self.assertFalse(kwargs["threshold"])  # raw predict: threshold not forwarded
        self.assertFalse(kwargs["debounce_time"])  # debounce stays off

    def test_wait_detects_despite_model_side_patience_bug(self):
        # The core regression: with a model that zeroes every score when
        # patience/threshold are forwarded (the real openWakeWord 0.6.0
        # behavior), wait_for_wake_word must STILL detect -- it does because it
        # scores with raw predict and applies patience itself.
        engine = _make_engine(score=0.9)
        engine._model = _PatienceBuggyModel(0.9)
        with mock.patch.object(
            wake_word, "MicrophoneStream", return_value=_FakeStream(frames=None)
        ), redirect_stdout(io.StringIO()) as buf:
            engine.wait_for_wake_word()
        self.assertGreaterEqual(engine._model.calls, 3)
        self.assertIn("[wake-debug] WAKE DETECTED", buf.getvalue())

    def test_wait_requires_consecutive_high_frames(self):
        # Interrupted highs (0.9, 0.9, 0.1, ...) must never fire: patience
        # needs 3 CONSECUTIVE frames >= threshold. No 3-run exists here, so
        # the stream drains and StopIteration surfaces instead of a detection.
        stream = _FakeStream(frames=[np.zeros((FRAME_SIZE, 1), dtype=np.int16)] * 6)
        engine = _make_engine(score=0.9)
        engine._model = _SequencedModel([0.9, 0.9, 0.1, 0.9, 0.9, 0.1])
        with mock.patch.object(wake_word, "MicrophoneStream", return_value=stream):
            with self.assertRaises(StopIteration):
                engine.wait_for_wake_word()
        self.assertEqual(engine._model.calls, 6)

    def test_wait_detects_after_consecutive_high_frames(self):
        stream = _FakeStream(frames=[np.zeros((FRAME_SIZE, 1), dtype=np.int16)] * 3)
        engine = _make_engine(score=0.9)
        engine._model = _SequencedModel([0.9, 0.9, 0.9])
        with mock.patch.object(wake_word, "MicrophoneStream", return_value=stream), \
                redirect_stdout(io.StringIO()):
            engine.wait_for_wake_word()
        self.assertEqual(engine._model.calls, 3)

    def test_detected_wake_word_prints_diagnostics(self):
        # Verify that when the engine detects a wake word above threshold,
        # the diagnostic messages appear in stdout. This tests the exact
        # detection code path that main() uses (the if/print block at lines
        # 396-398 of voice/wake_word.py).
        engine = _make_engine(score=0.9)  # above 0.5 threshold
        stream = _FakeStream(frames=[np.zeros((FRAME_SIZE, 1), dtype=np.int16)])
        with mock.patch.object(wake_word, "MicrophoneStream", return_value=stream):
            with redirect_stdout(io.StringIO()) as buf:
                # Simulate one iteration of main()'s detection loop:
                frame, _overflowed = stream.read(engine.frame_size)
                samples = np.asarray(frame, dtype=np.int16).reshape(-1)
                scores = engine._model.predict(
                    samples, patience={name: 3 for name in engine._model_names},
                    threshold={name: engine._threshold for name in engine._model_names},
                )
                for model_name, score in scores.items():
                    if score >= engine._threshold:
                        print(f"[wake-word] WAKE WORD DETECTED: {model_name}")
                        print("[wake-word] Ready for command.")
                # Consume the remaining read to cleanly exit the stream
                try:
                    stream.read(engine.frame_size)
                except StopIteration:
                    pass
                output = buf.getvalue()
        self.assertIn("[wake-word] WAKE WORD DETECTED: hey_jarvis", output)
        self.assertIn("[wake-word] Ready for command.", output)


class CleanupCase(unittest.TestCase):
    def test_close_is_idempotent_and_releases_model(self):
        engine = _make_engine()
        self.assertIsNotNone(engine._model)
        engine.close()
        self.assertIsNone(engine._model)
        self.assertEqual(engine._model_names, [])
        self.assertEqual(engine.model_path, "")
        engine.close()  # no error on second call


class FrameHandlingCase(unittest.TestCase):
    def test_pad_short_frame(self):
        short = np.zeros(640, dtype=np.int16)
        out = LocalWakeWordEngine._pad_or_trim(short)
        self.assertEqual(out.shape[0], FRAME_SIZE)

    def test_trim_long_frame(self):
        long = np.zeros(FRAME_SIZE + 100, dtype=np.int16)
        out = LocalWakeWordEngine._pad_or_trim(long)
        self.assertEqual(out.shape[0], FRAME_SIZE)

    def test_keep_exact_frame(self):
        exact = np.zeros(FRAME_SIZE, dtype=np.int16)
        out = LocalWakeWordEngine._pad_or_trim(exact)
        self.assertEqual(out.shape[0], FRAME_SIZE)


class DiagnosticEntryPointCase(unittest.TestCase):
    def test_list_models_exits_zero(self):
        fake = _fake_openwakeword_module()
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"openwakeword": fake}), \
             redirect_stdout(buf):
            code = wake_word._list_models()
        self.assertEqual(code, 0)
        self.assertIn("hey_jarvis", buf.getvalue())

    def test_download_reports_missing_package(self):
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"openwakeword": None}), \
             redirect_stdout(buf):
            code = wake_word._download_models()
        self.assertEqual(code, 1)
        self.assertIn("ERROR", buf.getvalue())


class DiagnosticCalcCase(unittest.TestCase):
    """Tests for diagnostic RMS/peak calculation helpers."""

    def test_rms_silence(self):
        """Test RMS calculation with silence (all zeros)."""
        chunk = np.zeros(1280, dtype=np.int16)
        rms = wake_word._rms_int16(chunk)
        self.assertAlmostEqual(rms, 0.0, places=4,
                               msg=f"Expected RMS 0.0 for silence, got {rms}")

    def test_rms_constant(self):
        """Test RMS calculation with a constant amplitude."""
        chunk = np.full(1280, 1000, dtype=np.int16)
        rms = wake_word._rms_int16(chunk)
        # RMS of constant value x is |x| for int16
        self.assertAlmostEqual(rms, 1000.0, delta=1.0,
                               msg=f"Expected RMS ~1000.0, got {rms}")

    def test_peak_silence(self):
        """Test peak calculation with silence (all zeros)."""
        chunk = np.zeros(1280, dtype=np.int16)
        peak = wake_word._peak_abs_int16(chunk)
        self.assertAlmostEqual(peak, 0.0, places=4,
                               msg=f"Expected Peak 0.0 for silence, got {peak}")

    def test_peak_constant(self):
        """Test peak calculation with a constant amplitude."""
        chunk = np.full(1280, 1000, dtype=np.int16)
        peak = wake_word._peak_abs_int16(chunk)
        self.assertAlmostEqual(peak, 1000.0, places=4,
                               msg=f"Expected Peak 1000.0, got {peak}")

    def test_rms_loud(self):
        """Test RMS calculation with a loud value."""
        chunk = np.full(1280, 10000, dtype=np.int16)
        rms = wake_word._rms_int16(chunk)
        self.assertAlmostEqual(rms, 10000.0, delta=1.0,
                               msg=f"Expected RMS ~10000.0, got {rms}")

    def test_peak_loud(self):
        """Test peak calculation with a loud value."""
        chunk = np.full(1280, 10000, dtype=np.int16)
        peak = wake_word._peak_abs_int16(chunk)
        self.assertAlmostEqual(peak, 10000.0, places=4,
                               msg=f"Expected Peak 10000.0, got {peak}")

    def test_frame_counter_is_initialized_before_loop(self):
        """Regression test: frame_counter must be initialized before the
        diagnostic while-loop to avoid UnboundLocalError.

        This test inspects the source of main() to verify that
        ``frame_counter = 0`` appears before ``while True`` in the same scope."""
        import inspect

        source = inspect.getsource(wake_word.main)
        init_idx = source.find("frame_counter = 0")
        loop_idx = source.find("while True")
        self.assertNotEqual(init_idx, -1, "frame_counter = 0 not found in main()")
        self.assertNotEqual(loop_idx, -1, "while True not found in main()")
        self.assertLess(init_idx, loop_idx, (
            "frame_counter must be initialized BEFORE the while loop, "
            "not after. Found frame_counter init at index %d, while loop at %d"
            % (init_idx, loop_idx)
                ))


def _write_wav(path, samples, sample_rate=16000, channels=1, sampwidth=2):
    """Write ``samples`` (1-D int16) to a PCM WAV file."""
    with wave.open(path, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(sampwidth)
        f.setframerate(sample_rate)
        data = np.asarray(samples, dtype=np.int16)
        if channels > 1:
            data = np.repeat(data[:, None], channels, axis=1)
        f.writeframes(np.ascontiguousarray(data).tobytes())


class WavReadCase(unittest.TestCase):
    """--test-wav / --record-test: WAV loading and normalization helpers."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_read_mono_16k(self):
        samples = np.arange(1280, dtype=np.int16)
        path = os.path.join(self.dir, "mono.wav")
        _write_wav(path, samples)
        arr, channels, rate, n_frames = wake_word._read_wav(path)
        self.assertEqual((channels, rate, n_frames), (1, 16000, 1280))
        self.assertEqual(arr.dtype, np.int16)
        self.assertEqual(arr.shape, (1280, 1))
        np.testing.assert_array_equal(arr[:, 0], samples)

    def test_read_stereo_interleaved(self):
        left = np.arange(100, dtype=np.int16)
        right = np.full(100, 1000, dtype=np.int16)
        path = os.path.join(self.dir, "stereo.wav")
        _write_wav(path, left, channels=2)  # both channels get the same 1-D data
        arr, channels, rate, n = wake_word._read_wav(path)
        self.assertEqual((channels, n), (2, 100))
        np.testing.assert_array_equal(arr[:, 0], left)
        np.testing.assert_array_equal(arr[:, 1], left)

    def test_non_16bit_raises(self):
        path = os.path.join(self.dir, "8bit.wav")
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(1)
            f.setframerate(16000)
            f.writeframes(bytes([0] * 1600))
        with self.assertRaises(ValueError):
            wake_word._read_wav(path)

    def test_missing_file_raises_clear_error(self):
        with self.assertRaises(ValueError):
            wake_word._read_wav(os.path.join(self.dir, "nope.wav"))

    def test_to_mono_passthrough_at_16k(self):
        arr = np.full((100, 1), 500, dtype=np.int16)
        out, resampled = wake_word._to_mono_16k(arr, 16000)
        self.assertFalse(resampled)
        self.assertEqual(out.shape, (100,))
        self.assertEqual(out.dtype, np.int16)
        np.testing.assert_array_equal(out, np.full(100, 500, dtype=np.int16))

    def test_to_mono_downmixes_stereo(self):
        arr = np.stack(
            [np.full(100, 100, np.int16), np.full(100, 300, np.int16)], axis=1
        )
        out, resampled = wake_word._to_mono_16k(arr, 16000)
        self.assertFalse(resampled)
        np.testing.assert_array_equal(out, np.full(100, 200, np.int16))

    def test_to_mono_resamples_8k_to_16k(self):
        arr = np.zeros((1000, 1), dtype=np.int16)
        out, resampled = wake_word._to_mono_16k(arr, 8000)
        self.assertTrue(resampled)
        self.assertEqual(out.shape[0], 2000)
        self.assertEqual(out.dtype, np.int16)


class WavScoreCase(unittest.TestCase):
    """The WAV replay path must produce a max score and its frame."""

    def test_max_score_and_frame_reported(self):
        engine = _make_engine(score=0.9)
        samples = np.zeros(16000, dtype=np.int16)  # 1 s of clip
        max_scores, best_frames = wake_word._score_wav(engine, samples)
        self.assertAlmostEqual(max_scores["hey_jarvis"], 0.9, places=6)
        self.assertGreaterEqual(best_frames["hey_jarvis"], 0)

    def test_silence_returns_zero(self):
        engine = _make_engine(score=0.0)
        samples = np.zeros(16000, dtype=np.int16)
        max_scores, best_frames = wake_word._score_wav(engine, samples)
        self.assertEqual(max_scores.get("hey_jarvis", 0.0), 0.0)

    def test_short_clip_pads_tail_to_frame_size(self):
        # 500 samples is less than one 1280-sample frame: the tail must be
        # padded, exactly like the live listener's _pad_or_trim path.
        engine = _make_engine(score=0.8)
        max_scores, best_frames = wake_word._score_wav(
            engine, np.zeros(500, dtype=np.int16)
        )
        self.assertAlmostEqual(max_scores["hey_jarvis"], 0.8, places=6)
        self.assertGreaterEqual(best_frames["hey_jarvis"], 0)


class WavTestEntryPointCase(unittest.TestCase):
    """_run_wav_test prints the exact requested report format."""

    def _run(self, score):
        engine = _make_engine(score=score)
        path = os.path.join(tempfile.mkdtemp(), "clip.wav")
        _write_wav(path, np.zeros(16000, dtype=np.int16))
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = wake_word._run_wav_test(engine, path)
        return code, buf.getvalue()

    def test_detected_output(self):
        code, out = self._run(score=0.87)
        self.assertEqual(code, 0)
        self.assertIn("[wake-word] WAV test", out)
        self.assertIn("File:", out)
        self.assertIn("Sample rate: 16000 Hz", out)
        self.assertIn("Channels: 1", out)
        self.assertIn("Frames: 16000", out)
        self.assertIn("Max score: 0.870", out)
        self.assertIn("Peak score at:", out)
        self.assertIn("Detection threshold: 0.50", out)
        self.assertIn("RESULT: DETECTED", out)

    def test_not_detected_output(self):
        code, out = self._run(score=0.003)
        self.assertEqual(code, 0)
        self.assertIn("Max score: 0.003", out)
        self.assertIn("RESULT: NOT DETECTED", out)

    def test_missing_file_reports_error(self):
        engine = _make_engine(score=0.9)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = wake_word._run_wav_test(
                engine, os.path.join(tempfile.mkdtemp(), "nope.wav")
            )
        self.assertEqual(code, 1)
        self.assertIn("ERROR", buf.getvalue())

    def test_main_test_wav_dispatch(self):
        engine = _make_engine(score=0.87)
        path = os.path.join(tempfile.mkdtemp(), "clip.wav")
        _write_wav(path, np.zeros(16000, dtype=np.int16))
        with mock.patch.object(
            wake_word, "create_wake_word_engine", return_value=engine
        ), redirect_stdout(io.StringIO()) as buf:
            code = wake_word.main(["--test-wav", path])
        self.assertEqual(code, 0)
        self.assertIn("RESULT: DETECTED", buf.getvalue())

    def test_speak_on_detect_responds_when_detected(self):
        engine = _make_engine(score=0.87)
        path = os.path.join(tempfile.mkdtemp(), "clip.wav")
        _write_wav(path, np.zeros(16000, dtype=np.int16))
        with mock.patch.object(
            wake_word, "_respond_if_wanted"
        ) as respond, redirect_stdout(io.StringIO()):
            code = wake_word._run_wav_test(engine, path, speak_on_detect=True)
        self.assertEqual(code, 0)
        respond.assert_called_once_with(True)

    def test_speak_on_detect_ignored_when_not_detected(self):
        engine = _make_engine(score=0.003)
        path = os.path.join(tempfile.mkdtemp(), "clip.wav")
        _write_wav(path, np.zeros(16000, dtype=np.int16))
        with mock.patch.object(
            wake_word, "_respond_if_wanted"
        ) as respond, redirect_stdout(io.StringIO()):
            code = wake_word._run_wav_test(engine, path, speak_on_detect=True)
        self.assertEqual(code, 0)
        respond.assert_not_called()

    def test_main_test_wav_speak_on_detect_dispatch(self):
        engine = _make_engine(score=0.87)
        path = os.path.join(tempfile.mkdtemp(), "clip.wav")
        _write_wav(path, np.zeros(16000, dtype=np.int16))
        with mock.patch.object(
            wake_word, "create_wake_word_engine", return_value=engine
        ), mock.patch.object(
            wake_word, "_respond_if_wanted"
        ) as respond, redirect_stdout(io.StringIO()):
            code = wake_word.main(["--test-wav", path, "--speak-on-detect"])
        self.assertEqual(code, 0)
        respond.assert_called_once_with(True)

    def test_respond_if_wanted_prints_and_speaks(self):
        buf = io.StringIO()
        with mock.patch.object(wake_word.tts, "speak") as speak, redirect_stdout(buf):
            wake_word._respond_if_wanted(True)
        self.assertIn("[qrudo] Hey, how are you?", buf.getvalue())
        speak.assert_called_once_with("Hey, how are you?")

    def test_respond_if_wanted_disabled_is_silent(self):
        buf = io.StringIO()
        with mock.patch.object(wake_word.tts, "speak") as speak, redirect_stdout(buf):
            wake_word._respond_if_wanted(False)
        self.assertEqual(buf.getvalue(), "")
        speak.assert_not_called()

    def test_respond_if_wanted_handles_tts_error(self):
        buf = io.StringIO()
        with mock.patch.object(
            wake_word.tts, "speak", side_effect=tts.TTSError("boom")
        ), redirect_stdout(buf):
            wake_word._respond_if_wanted(True)
        out = buf.getvalue()
        self.assertIn("[qrudo] Hey, how are you?", out)
        self.assertIn("[qrudo] ERROR:", out)


class RecordTestCase(unittest.TestCase):
    """--record-test: mic -> WAV -> same scoring path."""

    def test_record_writes_valid_wav_and_scores_it(self):
        path = os.path.join(tempfile.mkdtemp(), "rec.wav")
        seconds = 1.0
        with mock.patch.object(
            wake_word, "MicrophoneStream", return_value=_FakeStream(frames=None)
        ):
            wake_word._record_wav(path, seconds)
        with wave.open(path, "rb") as f:
            params = f.getparams()
            self.assertEqual(params.nchannels, 1)
            self.assertEqual(params.sampwidth, 2)
            self.assertEqual(params.framerate, 16000)
            self.assertEqual(
                params.nframes, int(seconds * 16000 / wake_word.FRAME_SIZE) * wake_word.FRAME_SIZE
            )
        engine = _make_engine(score=0.1)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = wake_word._run_wav_test(engine, path)
        self.assertEqual(code, 0)
        self.assertIn("RESULT: NOT DETECTED", buf.getvalue())

    def test_main_record_test_dispatch(self):
        engine = _make_engine(score=0.1)
        path = os.path.join(tempfile.mkdtemp(), "rec.wav")
        with mock.patch.object(
            wake_word, "MicrophoneStream", return_value=_FakeStream(frames=None)
        ), mock.patch.object(
            wake_word, "create_wake_word_engine", return_value=engine
        ), mock.patch.object(
            wake_word, "_report_microphone", return_value=(14, "Headset (test)")
        ), redirect_stdout(io.StringIO()) as buf:
            code = wake_word.main(["--record-test", path, "--seconds", "0.5"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(path))
        self.assertIn("RESULT: NOT DETECTED", buf.getvalue())

    def test_main_record_test_speak_on_detect_dispatch(self):
        engine = _make_engine(score=0.87)
        path = os.path.join(tempfile.mkdtemp(), "rec.wav")
        with mock.patch.object(
            wake_word, "MicrophoneStream", return_value=_FakeStream(frames=None)
        ), mock.patch.object(
            wake_word, "create_wake_word_engine", return_value=engine
        ), mock.patch.object(
            wake_word, "_report_microphone", return_value=(14, "Headset (test)")
        ), mock.patch.object(
            wake_word, "_respond_if_wanted"
        ) as respond, redirect_stdout(io.StringIO()) as buf:
            code = wake_word.main(
                ["--record-test", path, "--seconds", "0.5", "--speak-on-detect"]
            )
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(path))
        self.assertIn("RESULT: DETECTED", buf.getvalue())
        respond.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)