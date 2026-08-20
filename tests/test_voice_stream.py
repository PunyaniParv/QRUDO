"""Tests for voice/stream.py MicMonitor frame ownership.

Hardware-free: sounddevice is faked so the callback is fed directly.  These
tests pin down the single most important data-flow guarantee in the voice
stack: every frame a consumer keeps must be an independent copy, because
PortAudio reuses the SAME input buffer across callbacks.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from voice.stream import MicMonitor

_FRAME_SAMPLES = 1280


class _ReusingStream:
    """Fake sounddevice stream that mirrors PortAudio's buffer reuse.

    ``_buffer`` is one long-lived block; every callback hands the SAME view
    of it.  Between callbacks the "driver" overwrites the block with the next
    block of input, exactly like a real audio device.
    """

    def __init__(self, monitor, blocks):
        self._monitor = monitor
        self._blocks = list(blocks)
        self._buffer = np.zeros(_FRAME_SAMPLES, dtype=np.int16)

    def pump_all(self):
        for block in self._blocks:
            self._buffer[:] = block  # driver writes the next block
            self._monitor._callback(
                self._buffer, _FRAME_SAMPLES, None, None
            )
        return self

    def overwrite(self, value):
        self._buffer[:] = value  # simulate later input clobbering the block


class MicMonitorCallbackTest(unittest.TestCase):
    def test_callback_frames_survive_input_buffer_reuse(self):
        # THE regression: without .copy(), every queued frame is a view of the
        # same PortAudio block, so concatenating them later yields one block's
        # final contents -- captured speech silently becomes the last silence.
        monitor = MicMonitor(samplerate=16000, frame_samples=_FRAME_SAMPLES)
        stream = _ReusingStream(
            monitor,
            blocks=[
                np.full(_FRAME_SAMPLES, 1000, dtype=np.int16),
                np.full(_FRAME_SAMPLES, 2000, dtype=np.int16),
                np.full(_FRAME_SAMPLES, 3000, dtype=np.int16),
            ],
        )
        stream.pump_all()
        stream.overwrite(np.zeros(_FRAME_SAMPLES, dtype=np.int16))

        a = monitor.next_frame()
        b = monitor.next_frame()
        c = monitor.next_frame()
        self.assertEqual(a[0], 1000)
        self.assertEqual(b[0], 2000)
        self.assertEqual(c[0], 3000)
        self.assertIsNone(monitor.next_frame())

    def test_callback_frames_are_not_aliases_of_the_input(self):
        monitor = MicMonitor(samplerate=16000, frame_samples=_FRAME_SAMPLES)
        block = np.full(_FRAME_SAMPLES, 5000, dtype=np.int16)
        monitor._callback(block, _FRAME_SAMPLES, None, None)
        frame = monitor.next_frame()
        self.assertIsNot(frame, block)
        self.assertTrue(np.array_equal(frame, block))
        block[:] = 0  # caller mutating the source must not corrupt the frame
        self.assertEqual(frame[0], 5000)


if __name__ == "__main__":
    unittest.main()