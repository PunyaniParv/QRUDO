"""The idle saver, pinned from both sides.

The promise has two halves and both are load-bearing.  Energy: an
empty room must not pay for thirty inferences a second, so after the
grace the detector rests on alternate frames.  Performance: a hand in
frame is NEVER skipped -- gestures, swipes, dwell and the whole
detection floor run on every frame exactly as they always did, and
the only cost anywhere is that the first frame of a hand arriving
after long idle can land on a resting beat and be read one frame
(~33ms) later.  A change that breaks either half fails here.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from integration.runner import IdlePace

FRAME = 1.0 / 30


class TestAHandIsNeverSkipped(unittest.TestCase):
    def test_a_present_hand_keeps_full_rate_forever(self):
        """Frames with a hand in them are all detected, indefinitely."""

        pace = IdlePace()
        now = 0.0

        for _ in range(30 * 60):        # a full minute of hand
            self.assertFalse(pace.skip(now))
            pace.seen(now)              # the detector found it
            now += FRAME

    def test_the_whole_grace_period_runs_at_full_rate(self):
        """An empty frame inside the grace is still fully examined --
        a hand lost to blur for a moment must not meet a resting
        detector."""

        pace = IdlePace()
        now = 0.0
        pace.seen(now)

        while now - 0.0 <= pace.after:
            self.assertFalse(pace.skip(now))
            now += FRAME

    def test_one_detection_restores_full_rate_instantly(self):
        pace = IdlePace()
        now = 0.0
        pace.skip(now)                  # starts the idle clock

        now = pace.after + 5.0
        for _ in range(10):             # deep in the resting cadence
            pace.skip(now)
            now += FRAME

        pace.seen(now)                  # a hand arrives

        for _ in range(30):
            self.assertFalse(pace.skip(now))
            now += FRAME


class TestAnEmptyRoomRests(unittest.TestCase):
    def test_idle_rests_alternate_frames(self):
        pace = IdlePace()
        now = 0.0
        pace.skip(now)

        now = pace.after + 1.0
        skipped = 0
        for _ in range(30):
            if pace.skip(now):
                skipped += 1
            now += FRAME

        self.assertEqual(skipped, 15)   # exactly every other frame

    def test_arrival_after_idle_costs_at_most_one_frame(self):
        """Never two resting beats in a row: the worst a hand can meet
        is a single skipped frame before it is seen."""

        pace = IdlePace()
        now = 0.0
        pace.skip(now)

        now = pace.after + 1.0
        previous = False
        for _ in range(60):
            this = pace.skip(now)
            self.assertFalse(previous and this,
                             "two consecutive skips would double the "
                             "arrival cost the docstring promises")
            previous = this
            now += FRAME


if __name__ == "__main__":
    unittest.main(verbosity=2)
