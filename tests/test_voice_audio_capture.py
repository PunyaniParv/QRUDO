"""Tests for voice/audio_capture.py command capture and compatibility path.

Hardware-free: the monitor is faked with a scripted frame sequence. These
tests verify the adaptive-silence capture behaviour, the pre-roll seeding,
max-duration truncation, the no-speech fast path, and the record_until_silence
compatibility wrapper.
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voice import audio_capture

# 80 ms frames at 16 kHz (1280 samples) -- the monitor frame size.
_FRAME_SAMPLES = 1280
_LOUD = np.full(_FRAME_SAMPLES, 1000, dtype=np.int16)   # RMS 1000 >= threshold
_SILENT = np.zeros(_FRAME_SAMPLES, dtype=np.int16)


def _cfg(**overrides):
    base = {
        "frame_samples": _FRAME_SAMPLES,
        "sample_rate": 16000,
        "silence_duration_s": 0.7,
        "min_recording_s": 0.3,
        "max_command_s": 8.0,
        "pre_speech_timeout_s": 3.0,
        "silence_threshold_rms": 300.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeMonitor:
    """Fake MicMonitor replaying a scripted frame sequence."""

    def __init__(self, frames, pre_roll=None):
        self._frames = list(frames)
        self.pre_roll_frames = list(pre_roll or [])
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def next_frame(self, timeout=0.25, stop=None):
        if stop is not None and stop():
            return None
        if self._frames:
            return self._frames.pop(0)
        return None

    def pre_roll(self):
        return list(self.pre_roll_frames)

    def close(self):
        self.closed = True


class _Clock:
    """Scripted monotonic clock; each call advances by a frame interval."""

    def __init__(self, start=0.0, step=0.08):
        self.t = start
        self.step = step

    def __call__(self):
        t = self.t
        self.t += self.step
        return t


class CaptureCommandCase(unittest.TestCase):
    def test_ends_on_adaptive_silence_after_min_speech(self):
        # 5 loud frames (>= min_recording_s) then 12 silent frames
        # (>= silence_duration_s = 9 frames) ends the utterance.
        monitor = _FakeMonitor([_LOUD] * 5 + [_SILENT] * 12)
        audio = audio_capture.capture_command(monitor, config=_cfg())
        self.assertIsNotNone(audio)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape[0], 14 * _FRAME_SAMPLES)  # 5 speech + 9 quiet

    def test_prints_nothing_without_stats(self):
        monitor = _FakeMonitor([_LOUD] * 5 + [_SILENT] * 12)
        buf = io.StringIO()
        with redirect_stdout(buf):
            audio_capture.capture_command(monitor, config=_cfg())
        self.assertEqual(buf.getvalue(), "")

    def test_returns_stats_when_requested(self):
        monitor = _FakeMonitor([_LOUD] * 5 + [_SILENT] * 12)
        audio, stats = audio_capture.capture_command(monitor, config=_cfg(), stats=True)
        self.assertIsNotNone(audio)
        for key in ("first_speech_after", "duration", "final_silence", "samples"):
            self.assertIn(key, stats)
        self.assertEqual(stats["samples"], 14 * _FRAME_SAMPLES)
        self.assertAlmostEqual(stats["final_silence"], 0.72, places=2)

    def test_no_speech_returns_none(self):
        monitor = _FakeMonitor([_SILENT] * 5)  # quiet, then mic starves
        result = audio_capture.capture_command(monitor, config=_cfg())
        self.assertIsNone(result)

    def test_pre_roll_prepends_wake_tail(self):
        # The monitor's rolling memory (the wake phrase tail) is prepended so a
        # command in the same breath as the wake word is never clipped.
        monitor = _FakeMonitor(
            [_LOUD] * 5 + [_SILENT] * 12, pre_roll=[_LOUD, _LOUD]
        )
        audio = audio_capture.capture_command(monitor, config=_cfg())
        self.assertEqual(audio.shape[0], (2 + 14) * _FRAME_SAMPLES)

    def test_max_command_s_truncates_long_utterance(self):
        cfg = _cfg(max_command_s=0.3)  # ~3.75 frames -> truncate early
        monitor = _FakeMonitor([_LOUD] * 10)
        clock = _Clock()
        with mock.patch.object(audio_capture.time, "monotonic", clock):
            audio = audio_capture.capture_command(monitor, config=cfg)
        self.assertIsNotNone(audio)
        self.assertLess(audio.shape[0], 10 * _FRAME_SAMPLES)

    def test_stop_ends_capture_immediately(self):
        monitor = _FakeMonitor([_LOUD] * 5)
        result = audio_capture.capture_command(
            monitor, config=_cfg(), stop=lambda: True
        )
        self.assertIsNone(result)


class RecordUntilSilenceCompatibilityCase(unittest.TestCase):
    def _run(self, frames, report=False):
        monitor = _FakeMonitor(frames)
        buf = io.StringIO()
        with mock.patch.object(
            audio_capture, "MicMonitor", return_value=monitor
        ), redirect_stdout(buf):
            audio = audio_capture.record_until_silence(report=report)
        return audio, buf.getvalue(), monitor

    def test_compat_path_records_one_utterance(self):
        audio, out, monitor = self._run([_LOUD] * 5 + [_SILENT] * 12)
        self.assertEqual(out, "")
        self.assertEqual(audio.shape[0], 14 * _FRAME_SAMPLES)
        self.assertTrue(monitor.closed)

    def test_compat_path_returns_empty_on_no_speech(self):
        audio, out, monitor = self._run([_SILENT] * 5)
        self.assertEqual(audio.shape[0], 0)
        self.assertTrue(monitor.closed)

    def test_compat_path_report_prints_stats(self):
        audio, out, monitor = self._run([_LOUD] * 5 + [_SILENT] * 12, report=True)
        self.assertIn("[record-stats]", out)
        self.assertIn("first_speech_after=", out)
        self.assertIn("duration=", out)
        self.assertIn("final_silence=0.72s", out)
        self.assertIn(f"samples={14 * _FRAME_SAMPLES}", out)

    def test_compat_path_report_prints_no_speech(self):
        audio, out, monitor = self._run([_SILENT] * 5, report=True)
        self.assertIn("[record-stats]", out)
        self.assertIn("no_speech=yes", out)
        self.assertEqual(audio.shape[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)