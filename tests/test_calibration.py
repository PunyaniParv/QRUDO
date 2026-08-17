"""Turning recorded readings into thresholds.

The recording itself needs a camera and a person; this is the part that
decides where the lines go, which does not.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision import hand_state, motion
from vision.calibration import (Calibration, Profile, between, current,
                                from_samples, load_and_apply)


#: The names the app actually records readings under.  These were "imrp"
#: here, which no real reading has ever used -- so a threshold worked out
#: per finger passed its tests and found nothing on the day.
FINGERS = ("index", "middle", "ring", "pinky")


def readings(ext, reach, scale=0.15, count=20):
    """A pose held still, every finger reading the same."""

    return [{"ext": dict.fromkeys(FINGERS, ext),
             "reach": dict.fromkeys(FINGERS, reach),
             "scale": scale}
            for _ in range(count)]


def two_fingers(out=0.95, down=0.78, reach=1.55, scale=0.15, count=20):
    """Index and middle up, the other two just held down.

    Held down is not shut: a fist curls them to about 0.45 and this is
    0.78, which is most of the way to straight.  It is the case a
    threshold has to get right, and the one a fist does not show.
    """

    return [{"ext": {"index": out, "middle": out, "ring": down,
                     "pinky": down},
             "reach": dict.fromkeys(FINGERS, reach),
             "scale": scale}
            for _ in range(count)]


DEFAULTS = Calibration(0.82, 0.90, 1.15, 0.55, 0.30, 0.80, 0.60, 1.20, 0.60, 0.80, 0.035)


class TestTheReadingsAreKept(unittest.TestCase):
    """A calibration stores what was measured, not only what was concluded.

    Twice in one day the arithmetic that turns readings into thresholds
    turned out to be wrong, and both times the recording was gone -- so
    the only way to benefit from the correction was to record it again.
    Keeping the readings makes a correction free.
    """

    def setUp(self):
        # load_and_apply moves the live thresholds, and a test that moves
        # them must move them back: left applied, they followed every test
        # that ran after this file, and a detector test three modules away
        # quietly ran against this file's numbers instead of its own.
        self.before = current()

    def tearDown(self):
        self.before.apply()

    def poses(self):
        return {
            "fist": readings(0.45, 1.02),
            "open": readings(0.95, 1.60),
            "two": two_fingers(),
            "rest": readings(0.75, 1.40),
        }

    def moves(self):
        return {"turn left": [(0.90, 3.0), (0.85, 2.9)],
                "turn right": [(0.45, 1.6), (0.50, 1.8)],
                "raise": [(0.80, 2.2), (0.75, 2.0)],
                "lower": [(0.60, 1.7), (0.65, 1.8)]}

    def recorded_moves(self):
        """The shape the live recorder actually hands over: every peak a
        dict of both axes, which is what the crosstalk is measured from.
        The bare pairs above are the shape of older recordings, and for
        a while they were the only shape any test fed in -- so the suite
        stayed green while a real calibration crashed on its own
        recordings.
        """

        def rep(turn, speed, lift, lift_speed):
            return {"turn": turn, "speed": speed,
                    "lift": lift, "lift_speed": lift_speed}

        return {"turn left": [rep(0.90, 3.0, 0.10, 0.4),
                              rep(0.85, 2.9, 0.12, 0.5)],
                "turn right": [rep(0.45, 1.6, 0.08, 0.3),
                               rep(0.50, 1.8, 0.09, 0.4)],
                "raise": [rep(0.20, 0.7, 0.80, 2.2),
                          rep(0.15, 0.6, 0.75, 2.0)],
                "lower": [rep(0.18, 0.6, 0.60, 1.7),
                          rep(0.22, 0.7, 0.65, 1.8)]}

    def test_the_recorders_own_shape_survives_the_round_trip(self):
        profile = Profile.from_samples(self.poses(), self.recorded_moves())

        measured, _ = profile.derive(DEFAULTS)

        self.assertGreater(measured.swipe_lift, 0.0)
        self.assertIsNotNone(measured.crosstalk_turn)

        path = Path(tempfile.mkdtemp()) / "qrudo_calibration.json"
        measured.save(path, profile=profile)

        again, _ = Profile.load(path).derive(DEFAULTS)

        self.assertEqual(again, measured)

    def saved(self):
        """A calibration written to a temporary file, profile and all."""

        profile = Profile.from_samples(self.poses(), self.moves())
        measured, _ = profile.derive(DEFAULTS)

        path = Path(tempfile.mkdtemp()) / "qrudo_calibration.json"
        measured.save(path, profile=profile)

        return path, measured

    def test_what_was_measured_survives_the_round_trip(self):
        path, measured = self.saved()

        again, _ = Profile.load(path).derive(DEFAULTS)

        self.assertEqual(again, measured)

    def test_the_thresholds_are_still_written_at_the_top_level(self):
        """So a build from before profiles existed reads the file."""

        path, measured = self.saved()
        stored = json.loads(path.read_text())

        for name in Calibration.thresholds():
            self.assertIn(name, stored)

        self.assertEqual(stored["swipe_lift"], measured.swipe_lift)

    def test_a_file_without_readings_still_works(self):
        """Every calibration anyone has already done."""

        path = Path(tempfile.mkdtemp()) / "qrudo_calibration.json"
        path.write_text(json.dumps({
            "extended_ratio": 0.80, "open_ratio": 0.91, "fist_reach": 1.12,
            "swipe_turn": 0.40, "swipe_turn_speed": 1.70,
            "min_hand_on_screen": 0.03}) + "\n")

        self.assertIsNone(Profile.load(path))

        loaded = load_and_apply(path)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.extended_ratio, 0.80)

    def test_a_correction_reaches_a_session_already_recorded(self):
        """The whole point.  Nobody has to stand in front of a camera.

        Stood in for here by deriving the same readings against a
        different set of current values, which is what a correction to
        the arithmetic amounts to from the file's point of view.
        """

        path, _ = self.saved()
        profile = Profile.load(path)

        loose = Calibration(0.50, 0.50, 1.00, 0.55, 0.10, 0.10, 0.10, 0.10, 0.60, 0.80, 0.035)
        derived, _ = profile.derive(loose)

        self.assertGreater(derived.extended_ratio, 0.78)


class TestPoseAndMovementAreSeparate(unittest.TestCase):
    """So a gesture added later is a line, not another calibration.

    A gesture is a pose and a movement.  How straight your fingers go is
    one fact about you and how far your hand travels is another, and
    neither depends on the other -- so four fingers raised needs no new
    recording, only the two measurements this already holds.
    """

    def profile(self):
        return Profile.from_samples(
            {"fist": readings(0.45, 1.02), "open": readings(0.95, 1.60),
             "two": two_fingers(), "rest": readings(0.75, 1.40)},
            {"turn left": [(0.90, 3.0)], "turn right": [(0.45, 1.6)],
             "raise": [(0.80, 2.2)], "lower": [(0.60, 1.7)]})

    def test_the_vertical_movement_is_measured_at_last(self):
        """It never was.  Those two numbers decide every volume gesture."""

        measured, warnings = self.profile().derive(DEFAULTS)

        self.assertNotEqual(measured.swipe_lift, DEFAULTS.swipe_lift)
        self.assertFalse([w for w in warnings if "raise" in w])

    def test_the_weaker_direction_sets_the_vertical_bar_too(self):
        """An arm does not rise as far as it falls, or the other way."""

        measured, _ = self.profile().derive(DEFAULTS)

        self.assertLess(measured.swipe_lift, 0.60)

    def test_one_finger_is_enough_to_tell_a_resting_hand(self):
        """Reported as "it could not calculate two of them", and it was
        throwing a measurement away.

        A hand resting with a nearly straight index and a curled pinky
        separates perfectly well on the pinky.  Comparing the most
        extended resting finger against the least extended open one asks
        that *every* resting finger fall below the line, which is a
        stricter question than is_open asks -- it needs only one.
        """

        overlapping_index = [
            {"ext": {"index": 0.92, "middle": 0.91, "ring": 0.86,
                     "pinky": 0.76},
             "reach": dict.fromkeys(FINGERS, 1.40), "scale": 0.15}
            for _ in range(20)]

        spread = [
            {"ext": {"index": 0.91, "middle": 0.95, "ring": 0.96,
                     "pinky": 0.94},
             "reach": dict.fromkeys(FINGERS, 1.60), "scale": 0.15}
            for _ in range(20)]

        profile = Profile.from_samples(
            {"fist": readings(0.45, 1.02), "two": two_fingers(),
             "open": spread, "rest": overlapping_index}, {})

        measured, warnings = profile.derive(DEFAULTS)

        self.assertFalse([w for w in warnings if "resting one" in w])

        # Below every finger of the open hand, above one of the resting
        # hand: which is exactly what tells them apart.
        self.assertLess(measured.open_ratio, 0.91)
        self.assertGreater(measured.open_ratio, 0.76)

    def test_calibrating_close_up_says_the_range_did_not_change(self):
        """It is not a fault, and it is invisible in the numbers.

        The floor stays where it was, correctly -- a session recorded near
        the lens says nothing about what can be read across a room.  But
        somebody who calibrated in order to gain range has gained none.
        """

        near = Profile.from_samples(
            {"fist": readings(0.45, 1.02, scale=0.24)}, {})

        measured, warnings = near.derive(DEFAULTS)

        self.assertTrue(any("range is unchanged" in note
                            for note in measured.advice))

        # Advice, not a fault: nothing is guessed because of it, so it is
        # not among the things startup complains about.
        self.assertFalse([w for w in warnings if "range" in w])

    def test_open_never_ends_up_looser_than_a_single_finger(self):
        """The open-hand line is there to be the stricter of the two.

        They are measured against different poses, so they can come out
        the wrong way round -- and then a hand only just open enough for
        one test sails through the one meant to catch it.
        """

        measured, _ = self.profile().derive(DEFAULTS)

        self.assertGreaterEqual(measured.open_ratio, measured.extended_ratio)

    def test_not_recording_the_vertical_is_reported(self):
        profile = Profile.from_samples(
            {"fist": readings(0.45, 1.02)}, {"turn left": [(0.9, 3.0)]})

        measured, warnings = profile.derive(DEFAULTS)

        self.assertEqual(measured.swipe_lift, DEFAULTS.swipe_lift)
        self.assertTrue(any("raise or lower" in w for w in warnings))

    def test_four_fingers_raised_needs_nothing_new_recorded(self):
        """The pose is measured, the movement is measured; that is all
        such a gesture is."""

        profile = self.profile()

        self.assertTrue(profile.among(["open"], profile.LOW))
        self.assertTrue(profile.moves_like("raise", "lower"))


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
            "two": two_fingers(open_ext, scale=scale),
            "rest": readings(0.75, rest_reach, scale),
        }

    def test_a_line_sits_nearer_the_side_that_does_not_move(self):
        """A pose held for a prompt is tidier than one made in passing.

        Fingers folded right down for a camera come up in use; a finger
        that is simply straight stays straight.  Splitting the difference
        gives half the room to the side that does not need it, and the
        fist found out what that costs -- a line set from a hard clench
        stopped recognising ordinary fists altogether.
        """

        measured, _ = from_samples(self.poses(), {}, DEFAULTS)

        folded, straight = 0.78, 0.95      # what self.poses() holds
        middle = (folded + straight) / 2

        self.assertGreater(measured.extended_ratio, middle)
        self.assertLess(measured.extended_ratio, straight - 0.05)

    def test_the_same_for_a_fist(self):
        """Where the deliberate side is the fist rather than the fold."""

        measured, _ = from_samples(self.poses(), {}, DEFAULTS)

        self.assertGreater(measured.fist_reach, (1.02 + 1.40) / 2)
        self.assertGreater(measured.fist_curl, (0.45 + 0.75) / 2)

    def test_and_the_other_way_round_for_an_open_hand(self):
        """Here the deliberate side is the upper one -- a hand spread wide
        for a prompt is wider than one shown in passing -- so the line
        goes nearer the resting hand instead.

        Measured on a hand whose fingers fold well down, so that the
        finger line lands low and the guard keeping this above it does
        not decide the answer.
        """

        poses = dict(self.poses(), two=two_fingers(down=0.50))
        rest, spread = 0.75, 0.95

        measured, _ = from_samples(poses, {}, DEFAULTS)

        self.assertGreater(measured.open_ratio, rest)
        self.assertLess(measured.open_ratio, (rest + spread) / 2)

    def test_it_clears_two_fingers_held_down(self):
        """Reported as "it is not recognising 2 finger".

        Worked out from a fist against an open hand, the line landed at
        0.65 -- below where a ring finger sits when it is merely held
        down.  So a peace sign read as four fingers out and matched
        nothing at all.  The pose was being recorded the whole time and
        thrown away.
        """

        measured, _ = from_samples(self.poses(), {}, DEFAULTS)

        self.assertGreater(measured.extended_ratio, 0.78)

    def test_finger_threshold_lands_between_the_two(self):
        measured, warnings = from_samples(self.poses(), {}, DEFAULTS)

        self.assertGreater(measured.extended_ratio, 0.45)
        self.assertLess(measured.extended_ratio, 0.95)
        self.assertFalse([w for w in warnings if "held out" in w])

    def test_the_folded_line_lands_between_held_down_and_rest(self):
        """A second line under the first, drawn only where this hand's
        held-down fingers and its resting ones actually separate."""

        poses = self.poses()
        poses["two"] = two_fingers(down=0.55)
        poses["rest"] = readings(0.78, 1.40)

        measured, warnings = from_samples(poses, {}, DEFAULTS)

        self.assertGreater(measured.folded_ratio, 0.55)
        self.assertLess(measured.folded_ratio, 0.78)
        self.assertLess(measured.folded_ratio, measured.extended_ratio)
        self.assertFalse([w for w in warnings if "at rest" in w])

    def test_overlapping_recordings_keep_the_single_line_and_say_so(self):
        """This hand holds the swipe pose's spare fingers nearly straight
        -- above where its own rest reads -- and a gap invented between
        two overlapping measurements would cost it the pose.  Per
        finger: the ring and pinky are the ones that overlap here, and
        the ones the pose needs."""

        measured, warnings = from_samples(self.poses(), {}, DEFAULTS)

        self.assertEqual(measured.folded_ratios["ring"],
                         measured.extended_ratio)
        self.assertEqual(measured.folded_ratios["pinky"],
                         measured.extended_ratio)
        self.assertTrue([w for w in warnings if "at rest" in w])

    def test_fist_threshold_lands_between_fist_and_resting_hand(self):
        measured, _ = from_samples(self.poses(), {}, DEFAULTS)

        self.assertGreater(measured.fist_reach, 1.02)
        self.assertLess(measured.fist_reach, 1.40)

    def test_a_hand_that_never_opened_is_reported_not_guessed(self):
        """Somebody who held a fist throughout has told us nothing."""

        poses = self.poses(fist_ext=0.95)
        measured, warnings = from_samples(poses, {}, DEFAULTS)

        self.assertEqual(measured.extended_ratio, DEFAULTS.extended_ratio)
        self.assertTrue(any("held out" in w for w in warnings))

    def test_movement_thresholds_sit_under_the_weakest_attempt(self):
        """A gesture that only works when made emphatically will fail on
        the day, so the gentlest attempt sets the bar."""

        moves = {"turn": [(0.90, 3.0), (0.50, 2.0), (0.70, 2.5)]}
        measured, _ = from_samples(self.poses(), moves, DEFAULTS)

        self.assertLess(measured.swipe_turn, 0.50)
        self.assertLess(measured.swipe_turn_speed, 2.0)

    def test_the_harder_direction_sets_the_bar(self):
        """A wrist does not turn as far one way as the other.

        Reported as "swipe right is not detected".  It was being detected
        -- the threshold had been measured from a leftward turn, which goes
        further, and the rightward one was being asked for more than the
        joint had.
        """

        moves = {"turn left": [(0.90, 3.0), (0.85, 2.9)],
                 "turn right": [(0.45, 1.6), (0.50, 1.8)]}
        measured, warnings = from_samples(self.poses(), moves, DEFAULTS)

        self.assertLess(measured.swipe_turn, 0.45)
        self.assertLess(measured.swipe_turn_speed, 1.6)
        self.assertFalse([w for w in warnings if "wrist turn" in w])

    def test_one_direction_alone_is_still_measured(self):
        """An older calibration recorded a single "turn", and still counts."""

        moves = {"turn": [(0.60, 2.0), (0.60, 2.0)]}
        measured, warnings = from_samples(self.poses(), moves, DEFAULTS)

        self.assertLess(measured.swipe_turn, 0.60)
        self.assertFalse([w for w in warnings if "wrist turn" in w])

    def test_speed_is_given_more_room_than_size(self):
        """Asked to do the same thing four times, people vary how fast
        they do it more than how far.

        Measured on the recording that prompted this, and true of both
        movements on it.  One margin under the weakest attempt therefore
        leaves the speed bar too high -- a raise had to finish inside a
        third of a second, and one made at a normal pace never counted.
        """

        # Sizes all alike, speeds all over the place.
        moves = {"turn left": [(0.80, 6.0), (0.78, 2.4)],
                 "turn right": [(0.79, 5.0), (0.80, 3.0)]}

        measured, _ = from_samples(self.poses(), moves, DEFAULTS)

        weakest_size, weakest_speed = 0.78, 2.4

        self.assertAlmostEqual(measured.swipe_turn, weakest_size * 0.55, places=2)
        self.assertLess(measured.swipe_turn_speed, weakest_speed * 0.55)

    def test_repetitions_that_agree_are_taken_at_their_word(self):
        """Someone whose speed is as steady as their reach gets the same
        margin for both -- there is nothing to make room for."""

        moves = {"turn left": [(0.80, 4.0), (0.78, 3.9)],
                 "turn right": [(0.79, 3.95), (0.80, 4.0)]}

        measured, _ = from_samples(self.poses(), moves, DEFAULTS)

        self.assertAlmostEqual(measured.swipe_turn_speed, 3.9 * 0.55,
                               delta=0.05)

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


class TestTheRangeGateFollowsTheCamera(unittest.TestCase):
    """The one stored threshold that describes the window rather than
    the hand.  What decides whether a hand can be read is the pixels it
    covers in the original frame -- a property of the model -- so given
    the delivered width, the fraction follows, and the hand profile
    travels between cameras with nothing to redo.
    """

    def setUp(self):
        self.was = hand_state.MIN_HAND_ON_SCREEN

    def tearDown(self):
        hand_state.MIN_HAND_ON_SCREEN = self.was

    def test_agrees_with_the_measurement_the_default_came_from(self):
        """Poses were measured reliable down to 0.022 of frame at 640."""

        self.assertAlmostEqual(hand_state.set_camera(640), 0.022, places=3)

    def test_agrees_with_the_far_mode_measurement(self):
        """Range 'roughly doubles' at 1760: same pixels, half the
        fraction and then some."""

        self.assertLess(hand_state.set_camera(1760), 0.022 / 2)

    def test_a_strange_width_cannot_open_the_gate_to_noise(self):
        self.assertGreaterEqual(hand_state.set_camera(100000), 0.008)
        self.assertLessEqual(hand_state.set_camera(60), 0.10)

    def test_no_width_changes_nothing(self):
        before = hand_state.MIN_HAND_ON_SCREEN

        hand_state.set_camera(0)
        hand_state.set_camera(None)

        self.assertEqual(hand_state.MIN_HAND_ON_SCREEN, before)

    def test_the_camera_wins_over_the_stored_calibration(self):
        """The stored number was measured through the calibrating
        camera; the camera in use claims the gate afterwards, by
        arriving later -- which is the whole portability story."""

        calibration = current()
        calibration.min_hand_on_screen = 0.022
        calibration.apply()

        hand_state.set_camera(1600)

        self.assertLess(hand_state.MIN_HAND_ON_SCREEN, 0.022)


class TestApplying(unittest.TestCase):
    def setUp(self):
        self.before = current()

    def tearDown(self):
        self.before.apply()

    def test_apply_moves_the_thresholds_the_detectors_read(self):
        Calibration(0.7, 0.88, 1.3, 0.55, 0.2, 0.5, 0.5, 1.0, 0.60, 0.80, 0.02).apply()

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

    def test_a_file_missing_values_keeps_what_it_has(self):
        """An older file, from before a threshold existed.

        Discarding the whole thing meant a calibration somebody had sat
        down and done was silently ignored the next time one was added.
        """

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            path.write_text('{"extended_ratio": 0.8}')

            loaded = Calibration.load(path)

            assert loaded is not None, "an older file should still load"

            self.assertEqual(loaded.extended_ratio, 0.8)
            self.assertIn("open_ratio", loaded.incomplete)

    def test_a_file_of_nothing_useful_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            path.write_text('{"something": "else"}')
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
                "two": readings(0.95, 1.55), "rest": readings(0.75, 1.40),
}

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


class TestOpenHandThreshold(unittest.TestCase):
    """The line that keeps a resting hand from counting as an open one."""

    def poses(self, rest_ext=0.75):
        return {"fist": readings(0.45, 1.02), "open": readings(0.95, 1.60),
                "two": readings(0.95, 1.55), "rest": readings(rest_ext, 1.40),
}

    def test_lands_between_resting_and_open(self):
        measured, _ = from_samples(self.poses(), {}, DEFAULTS)

        self.assertGreater(measured.open_ratio, 0.75)
        self.assertLess(measured.open_ratio, 0.95)

    def test_sits_above_the_line_for_a_finger_being_out(self):
        """Open is a stronger claim than not-curled, so it asks for more."""

        measured, _ = from_samples(self.poses(), {}, DEFAULTS)

        # Equal when the measured open line would have fallen below it:
        # the guard raises it to meet the other rather than past it.
        self.assertGreaterEqual(measured.open_ratio, measured.extended_ratio)

    def test_a_slack_hand_that_looks_open_is_reported(self):
        measured, warnings = from_samples(self.poses(rest_ext=0.99), {},
                                          DEFAULTS)

        self.assertEqual(measured.open_ratio, DEFAULTS.open_ratio)
        self.assertTrue(any("resting" in w for w in warnings))


class TestImplausibleValuesArePulledBack(unittest.TestCase):
    """However it was measured, a threshold has to be usable.

    A measurement can be wrong -- a hand held badly, a landmark guessed at
    -- and a threshold far outside the plausible does not fail gently: it
    reads everything, or nothing, as the gesture.
    """

    def test_a_pinch_threshold_near_an_open_hand_is_pulled_in(self):
        wild = Calibration(0.82, 0.90, 1.15, 0.55, 0.30, 0.80, 0.60, 1.20, 0.60, 0.80, 0.90)

        kept, pulled = wild.sensible()

        self.assertLessEqual(kept.min_hand_on_screen, 0.12)
        self.assertTrue(any("min_hand_on_screen" in note for note in pulled))

    def test_sensible_values_are_left_alone(self):
        fine = Calibration(0.80, 0.92, 1.10, 0.55, 0.30, 0.80, 0.60, 1.20, 0.60, 0.80, 0.018)

        kept, pulled = fine.sensible()

        self.assertEqual(kept.min_hand_on_screen, 0.018)
        self.assertEqual(pulled, ())

    def test_the_range_floor_never_ends_up_stricter_than_shipped(self):
        """Reported as "it has stopped recognizing gestures".

        Calibrating near the lens measures a large hand and sets the floor
        under it, and everything past arm's length is then dropped before
        any gesture code sees it.  Nothing says so, because from where the
        user is standing a hand being discarded and a hand not being found
        look the same.

        Calibrating is allowed to reach further than the default.  It is
        not allowed to reach less far.
        """

        from vision import hand_state

        close_up = Calibration(0.80, 0.92, 1.10, 0.55, 0.30, 0.80, 0.60, 1.20, 0.60, 0.80, 0.0964)

        kept, pulled = close_up.sensible()

        self.assertEqual(kept.min_hand_on_screen, hand_state.MIN_HAND_ON_SCREEN)
        self.assertTrue(any("min_hand_on_screen" in note for note in pulled))

    def test_the_cap_is_the_shipped_default_itself(self):
        """So the two cannot drift apart unnoticed."""

        from vision import hand_state

        self.assertEqual(Calibration.BOUNDS["min_hand_on_screen"][1],
                         hand_state.MIN_HAND_ON_SCREEN)


class TestABadFrameOrTwo(unittest.TestCase):
    """A run of a hundred frames holds a few bad ones.

    Taken from a real calibration: it answered 0.10 for how straight a
    straight finger is, and 0.41 for how closed a fist is, when every
    honest frame said about 0.9 and 1.05.  Taking the very lowest and
    highest reading made the whole measurement hostage to the moment the
    hand was arriving, leaving, or half read.
    """

    def poses(self, bad=3):
        clean = readings(0.95, 1.60, count=100)
        rubbish = readings(0.12, 0.40, count=bad)

        return {"fist": readings(0.45, 1.02, count=100) + rubbish,
                "open": clean + rubbish,
                "two": clean,
                "rest": readings(0.75, 1.40, count=100)}

    def test_a_few_bad_frames_do_not_decide_it(self):
        measured, _ = from_samples(self.poses(), {}, DEFAULTS)

        self.assertGreater(measured.extended_ratio, 0.5)
        self.assertLess(measured.extended_ratio, 0.95)

    def test_the_fist_threshold_survives_them_too(self):
        measured, _ = from_samples(self.poses(), {}, DEFAULTS)

        self.assertGreater(measured.fist_reach, 0.9)
        self.assertLess(measured.fist_reach, 1.4)

    def test_a_clean_run_is_unaffected(self):
        measured, _ = from_samples(self.poses(bad=0), {}, DEFAULTS)

        self.assertGreater(measured.extended_ratio, 0.5)
        self.assertLess(measured.extended_ratio, 0.95)


class TestRangeOnlyLoosens(unittest.TestCase):
    """Calibrating close by must not shorten how far QRUDO can see.

    A hand is about a tenth of the frame across at a metre and a thirtieth
    at three.  Setting the floor from a calibration taken at the keyboard
    would put it at a hand's size there, and quietly stop the thing working
    from across the room -- which is the point of it.
    """

    def poses(self, scale):
        return {"fist": readings(0.45, 1.02, scale),
                "open": readings(0.95, 1.60, scale),
                "two": readings(0.95, 1.55, scale),
                "rest": readings(0.75, 1.40, scale)}

    def test_calibrating_close_by_keeps_the_range(self):
        measured, _ = from_samples(self.poses(0.25), {}, DEFAULTS)

        self.assertLessEqual(measured.min_hand_on_screen,
                             DEFAULTS.min_hand_on_screen)

    def test_calibrating_across_the_room_extends_it(self):
        measured, _ = from_samples(self.poses(0.04), {}, DEFAULTS)

        self.assertLess(measured.min_hand_on_screen,
                        DEFAULTS.min_hand_on_screen)
