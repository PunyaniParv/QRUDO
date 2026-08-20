"""Regression tests for the voice CLI output/logging UX (voice/log.py).

Verifies the NORMAL / DEBUG / TRACE separation end to end, hardware-free:

* NORMAL (no env):               only ``[voice]`` lifecycle lines, no frame
  telemetry, no repeated state messages while audio keeps flowing.
* DEBUG  (``QRUDO_VOICE_DEBUG``): USER + aggregate diagnostics
  (``[wake-word] WAKE WORD DETECTED``, ``[wake-stats]``, ``[record-stats]``,
  ``[whisper-stats]``, ``[voice-timing]``, ``[handoff-debug]``,
  ``[capture-debug]``) -- still no per-frame lines.
* TRACE  (``QRUDO_VOICE_DEBUG`` + ``QRUDO_VOICE_TRACE``): additionally emits
  ``[mic-frame]``, ``[mic-pop]`` and ``[capture-frame]``.

The gates are exercised both directly (voice/log.py) and through the real
call sites (``MicMonitor._callback``, ``capture_command``,
``run_voice_command_loop``).
"""

from __future__ import annotations

import importlib
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import unittest as _unittest

from control import Command
from voice import audio_capture
from voice import config as config_module
from voice import log as voice_log

# The pipeline imports faster_whisper at module load; without the
# voice extras this module SKIPS -- the suite must run green on both
# machines and both CI platforms.
try:
    from voice import pipeline
except ModuleNotFoundError as exc:
    raise _unittest.SkipTest(f"voice extras not installed: {exc}")
from voice.bridge import VoiceIntentRouter
from voice.stream import MicMonitor

# 80 ms frames at 16 kHz (1280 samples) -- the monitor frame size.
_FRAME_SAMPLES = 1280
_LOUD = np.full(_FRAME_SAMPLES, 1000, dtype=np.int16)
_SILENT = np.zeros(_FRAME_SAMPLES, dtype=np.int16)

_AUDIO = np.zeros(16000, dtype=np.float32)
_AUDIO_STATS = (_AUDIO, {
    "first_speech_after": 0.10,
    "duration": 0.30,
    "final_silence": 0.10,
    "samples": 12800,
})


def _cfg(**overrides):
    base = {
        "frame_samples": _FRAME_SAMPLES,
        "sample_rate": 16000,
        "silence_duration_s": 0.7,
        "min_recording_s": 0.3,
        "max_command_s": 8.0,
        "pre_speech_timeout_s": 3.0,
        "silence_threshold_rms": 300.0,
        "post_wake_grace_s": 0.8,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _patch_levels(debug: bool, trace: bool):
    """Point ``voice.config.CONFIG`` (which voice/log.py reads dynamically)
    at a level set.  ``VoiceConfig`` is frozen, so the module attribute is
    swapped for a plain namespace instead of setattr-ing the instance."""
    return mock.patch.object(
        config_module,
        "CONFIG",
        SimpleNamespace(
            debug=debug,
            trace=trace,
            sample_rate=16000,
            whisper_model_size="base",
        ),
    )


class _FakeMonitor:
    """Fake MicMonitor: never touches hardware, records lifecycle calls."""

    def __init__(self):
        self.opened = False
        self.closed = False
        self.drained = 0
        self._frames = []
        self.pre_roll_frames = []

    def open(self):
        self.opened = True

    def next_frame(self, timeout=0.25, stop=None, consumer=""):
        if self._frames:
            return self._frames.pop(0)
        return None

    def pre_roll(self):
        return list(self.pre_roll_frames)

    def pending_count(self):
        return len(self._frames)

    def drain(self):
        self.drained += 1
        self._frames.clear()
        self.pre_roll_frames.clear()

    def close(self):
        self.closed = True


class _FakeWake:
    name = "openwakeword"
    _model_names = ["hey_jarvis"]

    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.closed = False

    def initialize(self):
        pass

    def wait_for_wake_word(self, frame_source=None, stop=None, debug=False):
        self.calls += 1
        return True

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


class _FakeSTT:
    def __init__(self, transcript="volume up"):
        self.transcript = transcript
        self.model_name = "base"

    def transcribe(self, audio):
        return self.transcript


class _FakeEngine:
    def __init__(self):
        self.submitted = []
        self.sources = []
        self.payloads = []
        self.closed = False

    def submit(self, command, source="", payload=""):
        self.submitted.append(command)
        self.sources.append(source)
        self.payloads.append(payload)

    def close(self):
        self.closed = True


def _run_pipeline(stt="volume up", max_cycles=1, **kwargs):
    """Run run_voice_command_loop with hardware-free defaults; capture stdout."""
    capture = _AUDIO_STATS if kwargs.get("debug") else _AUDIO
    buf = io.StringIO()
    with mock.patch.object(
        pipeline, "capture_command", return_value=capture
    ), redirect_stdout(buf):
        pipeline.run_voice_command_loop(
            router=VoiceIntentRouter(),
            control_engine=_FakeEngine(),
            stt=_FakeSTT(stt),
            wake_engine=_FakeWake(),
            tts_speak=mock.Mock(),
            monitor=_FakeMonitor(),
            max_cycles=max_cycles,
            **kwargs,
        )
    return buf.getvalue()


# -- voice/log.py level gates -----------------------------------------

class LogLevelGateCase(unittest.TestCase):
    def test_normal_logs_lifecycle_only(self):
        with _patch_levels(debug=False, trace=False):
            buf = io.StringIO()
            with redirect_stdout(buf):
                voice_log.voice_log("[voice] Ready. Listening...")
                voice_log.voice_debug("[wake-stats]")
                voice_log.voice_trace("[mic-frame] seq=1 rms=10.0")
            out = buf.getvalue()
            self.assertIn("[voice] Ready. Listening...", out)
            self.assertNotIn("[wake-stats]", out)
            self.assertNotIn("[mic-frame]", out)

    def test_debug_adds_diagnostics_but_not_frames(self):
        with _patch_levels(debug=True, trace=False):
            buf = io.StringIO()
            with redirect_stdout(buf):
                voice_log.voice_debug("[wake-stats]")
                voice_log.voice_trace("[mic-frame] seq=1 rms=10.0")
            out = buf.getvalue()
            self.assertIn("[wake-stats]", out)
            self.assertNotIn("[mic-frame]", out)

    def test_trace_adds_frame_telemetry(self):
        with _patch_levels(debug=True, trace=True):
            buf = io.StringIO()
            with redirect_stdout(buf):
                voice_log.voice_debug("[wake-stats]")
                voice_log.voice_trace("[mic-frame] seq=1 rms=10.0")
            out = buf.getvalue()
            self.assertIn("[wake-stats]", out)
            self.assertIn("[mic-frame]", out)

    def test_trace_never_overrides_the_trace_half(self):
        # An explicit ``enabled`` can force the debug half but frame-level
        # output stays impossible unless CONFIG.trace is also on.
        with _patch_levels(debug=False, trace=False):
            buf = io.StringIO()
            with redirect_stdout(buf):
                voice_log.voice_trace("[mic-frame]", enabled=True)
            self.assertEqual(buf.getvalue(), "")


# -- MicMonitor frame telemetry ---------------------------------------

class MicFrameTraceCase(unittest.TestCase):
    def _monitor(self):
        mon = MicMonitor(frame_samples=_FRAME_SAMPLES, pre_roll_frames=16)
        return mon

    def test_no_frame_lines_in_normal_mode(self):
        with _patch_levels(debug=False, trace=False):
            mon = self._monitor()
            buf = io.StringIO()
            with redirect_stdout(buf):
                for _ in range(5):
                    mon._callback(_SILENT, _FRAME_SAMPLES, None, None)
                mon.next_frame(consumer="wake")
            self.assertEqual(buf.getvalue(), "")

    def test_no_frame_lines_in_debug_mode(self):
        with _patch_levels(debug=True, trace=False):
            mon = self._monitor()
            buf = io.StringIO()
            with redirect_stdout(buf):
                for _ in range(5):
                    mon._callback(_SILENT, _FRAME_SAMPLES, None, None)
                mon.next_frame(consumer="wake")
            self.assertEqual(buf.getvalue(), "")

    def test_frame_lines_in_trace_mode(self):
        with _patch_levels(debug=True, trace=True):
            mon = self._monitor()
            buf = io.StringIO()
            with redirect_stdout(buf):
                mon._callback(_LOUD, _FRAME_SAMPLES, None, None)
                mon.next_frame(consumer="wake")
            out = buf.getvalue()
            self.assertIn("[mic-frame]", out)
            self.assertIn("[mic-pop]", out)


# -- capture_command frame telemetry ----------------------------------

class CaptureFrameTraceCase(unittest.TestCase):
    def test_capture_frame_lines_only_in_trace(self):
        for debug, trace, present in ((False, False, False),
                                      (True, False, False),
                                      (True, True, True)):
            with self.subTest(debug=debug, trace=trace):
                with _patch_levels(debug=debug, trace=trace):
                    mon = MicMonitor(
                        frame_samples=_FRAME_SAMPLES, pre_roll_frames=16
                    )
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        for frame in [_LOUD] * 5 + [_SILENT] * 12:
                            mon._callback(frame, _FRAME_SAMPLES, None, None)
                        audio_capture.capture_command(
                            mon, config=_cfg(), debug=debug
                        )
                    out = buf.getvalue()
                    if present:
                        self.assertIn("[capture-frame]", out)
                    else:
                        self.assertNotIn("[capture-frame]", out)


# -- pipeline lifecycle (NORMAL/DEBUG/TRACE) --------------------------

class PipelineLogLevelCase(unittest.TestCase):
    def test_normal_shows_lifecycle_only(self):
        with _patch_levels(debug=False, trace=False):
            out = _run_pipeline()
            for line in (
                "[voice] Ready. Listening...",
                "[voice] Wake word detected.",
                "[voice] Listening for command...",
                '[voice] Command: "volume up"',
                "[voice] Executing: VOLUME_UP",
                "[voice] Done.",
            ):
                self.assertIn(line, out)
            for tag in (
                "[mic-frame]",
                "[mic-pop]",
                "[capture-frame]",
                "[wake-debug]",
                "[wake-stats]",
                "[handoff-debug]",
                "[stt-input-debug]",
                "[whisper-stats]",
                "[record-stats]",
                "[voice-timing]",
                "[capture-debug]",
            ):
                self.assertNotIn(tag, out)

    def test_debug_adds_diagnostics_but_never_frames(self):
        with _patch_levels(debug=True, trace=False):
            out = _run_pipeline(debug=True)
            for tag in (
                "[wake-word] WAKE WORD DETECTED: hey_jarvis",
                "[handoff-debug]",
                "[stt-input-debug]",
                "[whisper-stats]",
                "[record-stats]",
                "[voice-timing]",
            ):
                self.assertIn(tag, out)
            for tag in ("[mic-frame]", "[mic-pop]", "[capture-frame]"):
                self.assertNotIn(tag, out)

    def test_trace_pipeline_still_runs_lifecycle(self):
        with _patch_levels(debug=True, trace=True):
            out = _run_pipeline(debug=True)
            self.assertIn('[voice] Command: "volume up"', out)
            self.assertIn("[voice] Done.", out)

    def test_state_messages_not_repeated_per_frame(self):
        # Two wake->no-command cycles: each state transition prints its line
        # exactly once; frames flowing through the mic must not duplicate them.
        with _patch_levels(debug=False, trace=False):
            out = _run_pipeline(stt="", max_cycles=2)
            self.assertEqual(out.count("Wake word detected."), 2)
            self.assertEqual(out.count("[voice] No command detected."), 2)
            self.assertIn("[voice] Listening...", out)  # re-entry line
            for tag in ("[mic-frame]", "[mic-pop]", "[capture-frame]"):
                self.assertNotIn(tag, out)

    def test_capture_failure_recovers_without_double_state(self):
        with _patch_levels(debug=False, trace=False):
            buf = io.StringIO()
            with mock.patch.object(
                pipeline,
                "capture_command",
                side_effect=RuntimeError("mic went away"),
            ), redirect_stdout(buf):
                pipeline.run_voice_command_loop(
                    control_engine=_FakeEngine(),
                    stt=_FakeSTT("volume up"),
                    wake_engine=_FakeWake(),
                    tts_speak=mock.Mock(),
                    monitor=_FakeMonitor(),
                    max_cycles=2,
                )
            out = buf.getvalue()
            self.assertEqual(out.count("Wake word detected."), 2)
            self.assertEqual(
                out.count("[voice] ERROR: command capture failed: mic went away"), 2
            )
            self.assertNotIn("[mic-frame]", out)


class VoiceConfigTraceFlagCase(unittest.TestCase):
    def test_trace_flag_reads_qrudo_and_sarv_env(self):
        from voice import config as config_module

        def reload_with(**env):
            with mock.patch.dict("os.environ", env, clear=False):
                return importlib.reload(config_module)

        try:
            restored = dict(__import__("os").environ)
        except Exception:
            restored = {}
        try:
            config_module.CONFIG = reload_with(
                QRUDO_VOICE_TRACE="1", QRUDO_VOICE_DEBUG="1"
            ).CONFIG
            self.assertTrue(config_module.CONFIG.trace)
            self.assertTrue(config_module.CONFIG.debug)
            config_module.CONFIG = reload_with(
                SARV_VOICE_TRACE="1", QRUDO_VOICE_DEBUG=""
            ).CONFIG
            self.assertTrue(config_module.CONFIG.trace)
            config_module.CONFIG = reload_with(
                QRUDO_VOICE_DEBUG="1", QRUDO_VOICE_TRACE=""
            ).CONFIG
            self.assertTrue(config_module.CONFIG.debug)
            self.assertFalse(config_module.CONFIG.trace)
        finally:
            __import__("os").environ.clear()
            __import__("os").environ.update(
                {k: v for k, v in restored.items() if v is not None}
            )
            config_module.CONFIG = reload_with(
                **{k: v for k, v in restored.items() if v is not None}
            ).CONFIG


if __name__ == "__main__":
    unittest.main(verbosity=2)