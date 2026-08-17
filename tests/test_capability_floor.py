"""The floor: what QRUDO can do today, pinned so it can only improve.

The rest of the suite asks whether behaviours are right.  This file
asks a different question: has any measured capability got WORSE?  The
numbers below are today's actual measurements, taken deterministically
(seeded noise, a fake clock), and they are asserted as minimums -- so
a change that quietly trades recognition or speed away fails here,
loudly, before it can land.

The standing rule this file enforces: detection quality only ever goes
up.  A task that needs an edit gets its edit, but if this file goes
red, fixing that comes first and the task waits.  Improving a number
is welcome and shows up here as headroom; when it does, raise the
recorded floor to the new measurement -- deliberately, in its own
change -- so the improvement is locked in the same way.

The one deliberate exception is the open palm at the extreme edge of
range: it asks the most of the reading and is the shortest-ranged pose
by design, so its floor at the range gate is honest rather than high.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import vision
from vision import gestures, hand_state, motion

from test_gestures import (CURLED, EXTENDED, HAND, LOOSE, Clock, Point,
                           make_hand, peace_sign)

POSES = {
    "FIST": (CURLED, CURLED, CURLED, CURLED),
    "POINT": (EXTENDED, CURLED, CURLED, CURLED),
    "TWO_FINGER": (EXTENDED, EXTENDED, CURLED, CURLED),
    "OPEN_PALM": (EXTENDED, EXTENDED, EXTENDED, EXTENDED),
}

#: Two pixels of landmark error at the width the camera is asked for,
#: the same optimistic-but-honest noise the distance tests use.
BLUR = 2.0 / 1760

#: Recognition accuracy per pose per hand size, measured 2026-08-17.
#: Sizes are fractions of the frame width: 0.12 is arm's length, 0.035
#: about two metres, 0.022 the range gate itself.  Only ever raise
#: these.
ACCURACY_FLOOR = {
    "FIST":       {0.12: 1.0, 0.06: 1.0, 0.05: 1.0, 0.035: 1.0, 0.022: 1.0},
    "POINT":      {0.12: 1.0, 0.06: 1.0, 0.05: 1.0, 0.035: 1.0, 0.022: 1.0},
    "TWO_FINGER": {0.12: 1.0, 0.06: 1.0, 0.05: 1.0, 0.035: 1.0, 0.022: 1.0},
    "OPEN_PALM":  {0.12: 1.0, 0.06: 1.0, 0.05: 1.0, 0.035: 1.0, 0.022: 0.10},
}

#: How many frames of a standard 12-frame gesture pass before it
#: fires, measured 2026-08-17.  Latency is efficiency the user feels,
#: so it is floored too.  Only ever lower these.
PALM_RAISE_FIRES_BY = 8
TWO_FINGER_RAISE_FIRES_BY = 8
TURN_FIRES_BY = 4


def resized(points, wanted):
    wrist = points[0]
    scale = wanted / hand_state.hand_scale(points)

    return [Point(wrist.x + (point.x - wrist.x) * scale,
                  wrist.y + (point.y - wrist.y) * scale,
                  wrist.z + (point.z - wrist.z) * scale)
            for point in points]


def blurred(points, rng):
    return [Point(point.x + rng.gauss(0, BLUR),
                  point.y + rng.gauss(0, BLUR),
                  point.z + rng.gauss(0, BLUR))
            for point in points]


def accuracy(wanted, scale, trials=40, seed=3):
    """Deterministic: the same seed sees the same noise every run."""

    rng = random.Random(seed)
    base = resized(make_hand(*POSES[wanted]), scale)
    right = 0

    for _ in range(trials):
        vision.reset_state()

        for _ in range(6):
            got = gestures.detect_gesture(blurred(base, rng), HAND)

        right += got == wanted

    return right / trials


class FloorCase(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        motion.now = self.clock
        vision.reset_state()
        motion._state.cooldown_until = 0.0

    def tearDown(self):
        import time as real_time
        motion.now = real_time.time


class TestRecognitionNeverRegresses(FloorCase):
    def test_every_cell_of_the_accuracy_grid_holds(self):
        """Each pose, each distance, at least as right as today."""

        for pose, floors in ACCURACY_FLOOR.items():
            for scale, floor in floors.items():
                with self.subTest(pose=pose, scale=scale):
                    self.assertGreaterEqual(accuracy(pose, scale), floor)


class TestSpeedNeverRegresses(FloorCase):
    """A gesture that fires later than it used to has got worse, even
    if it still fires.  Frame indices of a standard 12-frame gesture,
    from a loose rest, exactly as a person makes one.
    """

    def rest_loose(self, cy=0.55):
        for _ in range(15):
            motion.detect_swipe(
                make_hand(LOOSE, LOOSE, LOOSE, LOOSE, cy=cy), HAND)
            self.clock.tick(0.5 / 15)

    def raise_fires_at(self, fingers):
        self.rest_loose()

        for i in range(12):
            cy = 0.55 + (0.35 - 0.55) * (i / 11)

            if motion.detect_swipe(make_hand(*fingers, cy=cy), HAND):
                return i

            self.clock.tick(0.30 / 11)

        return 99

    def test_a_palm_raise_fires_no_later_than_today(self):
        self.assertLessEqual(self.raise_fires_at(POSES["OPEN_PALM"]),
                             PALM_RAISE_FIRES_BY)

    def test_a_two_finger_raise_fires_no_later_than_today(self):
        self.assertLessEqual(self.raise_fires_at(POSES["TWO_FINGER"]),
                             TWO_FINGER_RAISE_FIRES_BY)

    def test_a_turn_fires_no_later_than_today(self):
        for _ in range(8):
            motion.detect_swipe(peace_sign(roll=math.radians(-25)), HAND)
            self.clock.tick(0.25 / 6)

        for i in range(12):
            if motion.detect_swipe(
                    peace_sign(roll=math.radians(-25 + 50 * (i / 11))), HAND):
                self.assertLessEqual(i, TURN_FIRES_BY)
                return

            self.clock.tick(0.30 / 11)

        self.fail("the turn never fired at all")


if __name__ == "__main__":
    unittest.main(verbosity=2)
