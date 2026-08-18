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
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

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
        self.models = {"hey_jarvis": object()}

    def predict(self, samples, patience=None, threshold=None, debounce_time=0.0):
        self.calls += 1
        assert samples.shape[0] == FRAME_SIZE
        return {"hey_jarvis": self.score}


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
        for attr in ("initialize", "wait_for_wake_word", "close"):
            self.assertTrue(callable(getattr(engine, attr)))
        self.assertEqual(engine.name, "openwakeword")
        self.assertEqual(engine.sample_rate, 16000)
        self.assertEqual(engine.frame_size, 1280)


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
        ):
            engine.wait_for_wake_word()
        self.assertEqual(engine._model.calls, 1)

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
        with mock.patch.object(wake_word, "MicrophoneStream", return_value=stream):
            engine.wait_for_wake_word()
        self.assertEqual(stream.entered, 1)
        self.assertEqual(stream.exited, 1)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)