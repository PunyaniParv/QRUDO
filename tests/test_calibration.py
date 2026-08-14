"""Turning recorded readings into thresholds.

The recording itself needs a camera and a person; this is the part that
decides where the lines go, which does not.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision import hand_state, motion
from vision.calibration import Calibration, between, current, from_samples


def readings(ext, reach, scale=0.15, count=20):
    """A pose held still, every finger reading the same."""

    return [{"ext": dict.fromkeys("imrp", ext),
             "reach": dict.fromkeys("imrp", reach),
             "scale": scale}
            for _ in range(count)]


DEFAULTS = Calibration(0.82, 1.15, 0.30, 0.80, 0.60, 1.20, 0.035)


class TestWhereTheLineGoes(unittest.TestCase):
    def test_between_two_measurements(self):
        line, ok = between(0.4, 0.8, defaults_to=99)
        self.assertTrue(ok)
        self.assertAlmostEqual(line, 0.6)

    def test_overlapping_measurements_keep_the_old_value(self):
        """If the two cases measured the same, there is no line to draw."""

        line, ok = between(0.9, 0.5, defaults_to=0.82)
        self.assertFalse(ok)
        self.assertEqual(line, 0.82)


class TestFromSamples(unittest.TestCase):
    def poses(self, fist_ext=0.45, open_ext=0.95, fist_reach=1.02,
              rest_reach=1.40, scale=0.15):
        return {
            "fist": readings(fist_ext, fist_reach, scale),
            "open": readings(open_ext, 1.60, scale),
            "two": readings(open_ext, 1.55, scale),
            "rest": readings(0.75, rest_reach, scale),
        }

    def test_finger_threshold_lands_between_the_two(self):
        measured, warnings = from_samples(self.poses(), {}, DEFAULTS)

        self.assertGreater(measured.extended_ratio, 0.45)
        self.assertLess(measured.extended_ratio, 0.95)
        self.assertFalse([w for w in warnings if "finger" in w])

    def test_fist_threshold_lands_between_fist_and_resting_hand(self):
        measured, _ = from_samples(self.poses(), {}, DEFAULTS)

        self.assertGreater(measured.fist_reach, 1.02)
        self.assertLess(measured.fist_reach, 1.40)

    def test_a_hand_that_never_opened_is_reported_not_guessed(self):
        """Somebody who held a fist throughout has told us nothing."""

        poses = self.poses(fist_ext=0.95)
        measured, warnings = from_samples(poses, {}, DEFAULTS)

        self.assertEqual(measured.extended_ratio, DEFAULTS.extended_ratio)
        self.assertTrue(any("straight finger" in w for w in warnings))

    def test_movement_thresholds_sit_under_the_weakest_attempt(self):
        """A gesture that only works when made emphatically will fail on
        the day, so the gentlest attempt sets the bar."""

        moves = {"turn": [(0.90, 3.0), (0.50, 2.0), (0.70, 2.5)]}
        measured, _ = from_samples(self.poses(), moves, DEFAULTS)

        self.assertLess(measured.swipe_turn, 0.50)
        self.assertLess(measured.swipe_turn_speed, 2.0)

    def test_no_movement_recorded_is_reported(self):
        measured, warnings = from_samples(self.poses(), {}, DEFAULTS)

        self.assertEqual(measured.swipe_turn, DEFAULTS.swipe_turn)
        self.assertTrue(any("wrist turn" in w for w in warnings))

    def test_range_follows_where_you_stood(self):
        """Calibrated across the room, a small hand still has to count."""

        far, _ = from_samples(self.poses(scale=0.05), {}, DEFAULTS)
        near, _ = from_samples(self.poses(scale=0.25), {}, DEFAULTS)

        self.assertLess(far.min_hand_on_screen, near.min_hand_on_screen)
        self.assertLess(far.min_hand_on_screen, 0.05)


class TestApplying(unittest.TestCase):
    def setUp(self):
        self.before = current()

    def tearDown(self):
        self.before.apply()

    def test_apply_moves_the_thresholds_the_detectors_read(self):
        Calibration(0.7, 1.3, 0.2, 0.5, 0.4, 0.9, 0.02).apply()

        self.assertEqual(hand_state.EXTENDED_RATIO, 0.7)
        self.assertEqual(hand_state.FIST_REACH, 1.3)
        self.assertEqual(motion.SWIPE_TURN, 0.2)
        self.assertEqual(hand_state.MIN_HAND_ON_SCREEN, 0.02)

    def test_saved_and_loaded_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            DEFAULTS.save(path)
            self.assertEqual(Calibration.load(path), DEFAULTS)

    def test_missing_file_is_not_an_error(self):
        self.assertIsNone(Calibration.load(Path("/nowhere/at/all.json")))

    def test_a_damaged_file_is_ignored_rather_than_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            path.write_text("{ this is not json")
            self.assertIsNone(Calibration.load(path))

    def test_a_file_missing_values_is_ignored(self):
        """An older file from a version with fewer thresholds."""

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            path.write_text('{"extended_ratio": 0.8}')
            self.assertIsNone(Calibration.load(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestABotchedAttempt(unittest.TestCase):
    """One missed repetition must not set the bar on the floor.

    Seen for real: a prompt arrived while the hand was still moving back,
    that attempt measured almost nothing, and it would have set the wrist
    threshold to 0.017 -- low enough to fire at a resting hand.
    """

    def poses(self):
        return {"fist": readings(0.45, 1.02), "open": readings(0.95, 1.60),
                "two": readings(0.95, 1.55), "rest": readings(0.75, 1.40)}

    def test_an_attempt_that_barely_moved_is_ignored(self):
        moves = {"turn": [(0.03, 0.19), (0.93, 3.85), (0.53, 2.40)]}
        measured, warnings = from_samples(self.poses(), moves, DEFAULTS)

        self.assertGreater(measured.swipe_turn, 0.2)
        self.assertTrue(any("barely moved" in w for w in warnings))

    def test_gentle_but_real_attempts_still_count(self):
        """Consistently gentle is a person, not a mistake."""

        moves = {"turn": [(0.50, 2.0), (0.45, 1.9), (0.55, 2.2)]}
        measured, warnings = from_samples(self.poses(), moves, DEFAULTS)

        self.assertLess(measured.swipe_turn, DEFAULTS.swipe_turn)
        self.assertFalse([w for w in warnings if "barely" in w])

    def test_nothing_lands_below_the_floor(self):
        """Even if every attempt was tiny, the result stays usable."""

        moves = {"turn": [(0.02, 0.1), (0.02, 0.1), (0.02, 0.1)]}
        measured, _ = from_samples(self.poses(), moves, DEFAULTS)

        self.assertGreaterEqual(measured.swipe_turn,
                                DEFAULTS.swipe_turn * 0.4)
