"""Tests for integration/voice.py -- the glue that runs voice beside the camera.

The voice stack is optional, so these tests fake its absence and its
presence, and the pipeline call itself is faked: what matters here is
that a requested voice session starts on a thread sharing the engine,
tags its commands with a voice source, and shuts down cleanly -- and
that a machine without the voice requirements is exactly the
camera-only app it was before.
"""

from __future__ import annotations

import io
import subprocess
import sys
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from integration import voice as voice_mod


class TestAvailable(unittest.TestCase):
    def test_true_when_every_dependency_is_present(self):
        with mock.patch("importlib.util.find_spec", return_value=object()):
            self.assertTrue(voice_mod.available())

    def test_false_when_a_dependency_is_missing(self):
        def find_spec(name, **kwargs):
            return None if name == "openwakeword" else object()

        with mock.patch("importlib.util.find_spec", side_effect=find_spec):
            self.assertFalse(voice_mod.available())

    def test_false_when_nothing_is_installed(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(voice_mod.available())


class TestStart(unittest.TestCase):
    def test_start_returns_none_without_the_voice_stack(self):
        engine = mock.Mock()

        with mock.patch.object(voice_mod, "available", return_value=False), \
                redirect_stdout(io.StringIO()) as buf:
            handle = voice_mod.start(engine)

        self.assertIsNone(handle)
        self.assertIn("requirements-voice.txt", buf.getvalue())
        engine.assert_not_called()

    def test_start_runs_the_loop_on_a_thread_with_the_shared_engine(self):
        engine = mock.Mock()
        seen = {}

        def fake_run(engine, source, stop_event):
            seen["engine"] = engine
            seen["source"] = source
            while not stop_event.is_set():
                time.sleep(0.01)

        with mock.patch.object(voice_mod, "available", return_value=True), \
                mock.patch.object(voice_mod, "_run", side_effect=fake_run):
            handle = voice_mod.start(engine)

        self.assertIsNotNone(handle)
        self.assertTrue(handle.running)
        self.assertIs(seen["engine"], engine)
        self.assertEqual(seen["source"], "voice")

        handle.stop(timeout=5.0)

        self.assertFalse(handle.running)

    def test_start_owns_a_fresh_stop_event(self):
        engine = mock.Mock()

        with mock.patch.object(voice_mod, "available", return_value=True), \
                mock.patch.object(voice_mod, "_run",
                                  side_effect=lambda _e, _s, se: se.wait(5.0)):
            handle = voice_mod.start(engine)

        self.assertIsNotNone(handle)
        self.assertTrue(handle.running)

        handle.stop(timeout=5.0)

        self.assertFalse(handle.running)

    def test_a_caller_supplied_stop_event_is_used(self):
        engine = mock.Mock()
        shared = threading.Event()
        used = []

        def fake_run(_e, _s, stop_event):
            used.append(stop_event)
            return stop_event.wait(2.0)

        with mock.patch.object(voice_mod, "available", return_value=True), \
                mock.patch.object(voice_mod, "_run", side_effect=fake_run):
            handle = voice_mod.start(engine, stop_event=shared)

        self.assertIsNotNone(handle)
        self.assertIs(used[0], shared)
        shared.set()
        handle.thread.join(timeout=5.0)
        self.assertFalse(handle.running)


class TestRunIsBestEffort(unittest.TestCase):
    """The voice thread can never take the camera loop down with it."""

    def test_run_hands_the_engine_source_and_stop_to_the_pipeline(self):
        engine = mock.Mock()
        stop = threading.Event()
        seen = {}

        def fake_loop(*, control_engine, source, should_stop, **kwargs):
            seen["engine"] = control_engine
            seen["source"] = source
            seen["stop"] = should_stop

        import voice.pipeline as pipeline

        with mock.patch.object(pipeline, "run_voice_command_loop",
                               side_effect=fake_loop):
            voice_mod._run(engine, "voice", stop)

        self.assertIs(seen["engine"], engine)
        self.assertEqual(seen["source"], "voice")
        self.assertEqual(seen["stop"], stop.is_set)

    def test_an_unexpected_pipeline_failure_is_printed_not_raised(self):
        engine = mock.Mock()
        stop = threading.Event()

        import voice.pipeline as pipeline

        with mock.patch.object(
                pipeline, "run_voice_command_loop",
                side_effect=RuntimeError("mic went away")), \
                redirect_stdout(io.StringIO()) as buf:
            voice_mod._run(engine, "voice", stop)

        self.assertIn("voice: stopped unexpectedly", buf.getvalue())
        self.assertIn("mic went away", buf.getvalue())

    def test_a_missing_voice_module_is_reported(self):
        engine = mock.Mock()

        with mock.patch.dict(sys.modules, {"voice.pipeline": None}), \
                redirect_stdout(io.StringIO()) as buf:
            voice_mod._run(engine, "voice", threading.Event())

        self.assertIn("voice: cannot start", buf.getvalue())


class TestHandle(unittest.TestCase):
    def test_stop_is_safe_when_the_thread_has_already_gone(self):
        stop = threading.Event()
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()

        voice_mod._VoiceHandle(thread, stop).stop()


class TestRunVoiceOnly(unittest.TestCase):
    """The standalone assistant: --voice with no camera mode never touches
    the camera, and shuts down the way the gesture loop does."""

    def make_engine(self):
        engine = mock.Mock()
        engine.controller.name = "null"
        engine.config.dry_run = False
        return engine

    def test_runs_until_asked_to_stop_then_closes_everything(self):
        engine = self.make_engine()
        stop = iter([False, True])

        with mock.patch.object(voice_mod, "available", return_value=True), \
                mock.patch.object(voice_mod, "start") as start, \
                redirect_stdout(io.StringIO()) as buf:
            code = voice_mod.run_voice_only(engine,
                                            should_stop=lambda: next(stop))

        self.assertEqual(code, 0)
        start.assert_called_once_with(engine)
        start.return_value.stop.assert_called_once_with()
        engine.close.assert_called_once_with()
        self.assertIn("QRUDO voice", buf.getvalue())
        self.assertIn("bye.", buf.getvalue())

    def test_missing_voice_stack_returns_1_and_never_starts(self):
        engine = self.make_engine()

        with mock.patch.object(voice_mod, "available", return_value=False), \
                mock.patch.object(voice_mod, "start") as start, \
                redirect_stdout(io.StringIO()) as buf:
            code = voice_mod.run_voice_only(engine)

        self.assertEqual(code, 1)
        start.assert_not_called()
        engine.close.assert_called_once_with()
        self.assertIn("requirements-voice.txt", buf.getvalue())

    def test_a_keyboard_interrupt_is_a_clean_shutdown(self):
        engine = self.make_engine()
        calls = []

        def stop_when_asked():
            if not calls:
                calls.append("stopping")
                raise KeyboardInterrupt
            return False

        with mock.patch.object(voice_mod, "available", return_value=True), \
                mock.patch.object(voice_mod, "start") as start, \
                redirect_stdout(io.StringIO()):
            code = voice_mod.run_voice_only(engine, should_stop=stop_when_asked)

        self.assertEqual(code, 0)
        start.return_value.stop.assert_called_once_with()
        engine.close.assert_called_once_with()

    def test_voice_is_stopped_before_the_engine_closes(self):
        engine = self.make_engine()
        order = []
        engine.close = lambda *a, **k: order.append("engine")

        start = mock.Mock()
        start.return_value.stop = lambda *a, **k: order.append("voice")

        with mock.patch.object(voice_mod, "available", return_value=True), \
                mock.patch.object(voice_mod, "start", return_value=start.return_value), \
                redirect_stdout(io.StringIO()):
            voice_mod.run_voice_only(engine, should_stop=lambda: True)

        self.assertEqual(order, ["voice", "engine"])

    def test_dry_run_is_announced(self):
        engine = self.make_engine()
        engine.config.dry_run = True

        with mock.patch.object(voice_mod, "available", return_value=True), \
                mock.patch.object(voice_mod, "start"), \
                redirect_stdout(io.StringIO()) as buf:
            voice_mod.run_voice_only(engine, should_stop=lambda: True)

        self.assertIn("DRY RUN", buf.getvalue())


class TestVoiceNeverTouchesCamera(unittest.TestCase):
    """The voice-only mode must never initialize the camera stack."""

    def test_importing_the_voice_stack_never_imports_camera_modules(self):
        code = (
            "import sys; "
            "import integration.voice, voice.pipeline, voice.wake_word, "
            "voice.stream, voice.detect, voice.audio_capture; "
            "assert 'cv2' not in sys.modules, 'cv2 imported by the voice stack'; "
            "assert 'mediapipe' not in sys.modules, "
            "'mediapipe imported by the voice stack'; "
            "print('VOICE_STACK_OK')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(ROOT),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VOICE_STACK_OK", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
