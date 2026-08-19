"""Tests for voice/audio_capture.py recording instrumentation.

Hardware-free: MicrophoneStream is faked with a scripted frame sequence. These
tests only verify that the ``report=True`` measurement output is printed and
that the default path (``report=False``) behaves exactly as before -- no
recording behavior is changed by the instrumentation.
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voice import audio_capture

# 30 ms chunks at 16 kHz, shaped like a sounddevice read on a mono stream.
_CHUNK_SAMPLES = 480
_LOUD = np.full((_CHUNK_SAMPLES, 1), 1000, dtype=np.int16)   # RMS 1000 >= 300
_SILENT = np.zeros((_CHUNK_SAMPLES, 1), dtype=np.int16)


class _FakeStream:
    """A fake MicrophoneStream playing back a fixed frame sequence."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc_info):
        self.exited += 1
        return False

    def read(self, frames):
        if not self._frames:
            raise StopIteration
        return self._frames.pop(0), False


class RecordUntilSilenceCase(unittest.TestCase):
    def _run(self, frames, report):
        stream = _FakeStream(frames)
        buf = io.StringIO()
        with mock.patch.object(
            audio_capture, "MicrophoneStream", return_value=stream
        ), redirect_stdout(buf):
            audio = audio_capture.record_until_silence(report=report)
        return audio, buf.getvalue(), stream

    def test_default_path_is_silent(self):
        # 10 loud + 30 silent chunks ends the recording (min 0.3 s + 0.9 s quiet).
        frames = [_LOUD] * 10 + [_SILENT] * 30
        audio, out, stream = self._run(frames, report=False)
        self.assertEqual(out, "")
        self.assertEqual(audio.shape[0], 40 * _CHUNK_SAMPLES)
        self.assertEqual(stream.entered, 1)
        self.assertEqual(stream.exited, 1)

    def test_report_prints_stats_and_keeps_audio_identical(self):
        frames = [_LOUD] * 10 + [_SILENT] * 30
        audio, out, stream = self._run(frames, report=True)
        self.assertIn("[record-stats]", out)
        self.assertIn("first_speech_after=", out)
        self.assertIn("duration=", out)
        self.assertIn("final_silence=0.90s", out)
        self.assertIn(f"samples={40 * _CHUNK_SAMPLES}", out)
        self.assertEqual(audio.shape[0], 40 * _CHUNK_SAMPLES)
        self.assertEqual(stream.entered, 1)
        self.assertEqual(stream.exited, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
