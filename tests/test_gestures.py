"""Gesture detection, tested with synthetic hands at every angle.

No camera is involved.  A hand is built in 3D -- knuckles, joints,
fingertips -- and then rotated to whatever viewpoint a test wants, so the
same fist can be shown to the camera face-on, in three-quarter view, in
side profile, or from behind.

What this proves is that the maths does not care which way the hand is
turned.  What it cannot prove is that MediaPipe still reports accurate
landmarks at those angles: these hands are geometrically perfect, and a
real one seen edge-on is partly hidden from the lens.  Extreme angles will
always be harder in practice than they are here.

The frame is mirrored before detection, so MediaPipe labels a physical
right hand "Left".  These hands are built accordingly.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import vision
from vision import gestures, hand_state, motion


class Point:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Clock:
    """Stand-in for time.time() so motion can be replayed deterministically."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


# --- building a hand -------------------------------------------------------

EXTENDED = "extended"
CURLED = "curled"      # shut: fingertips back at the knuckles
LOOSE = "loose"        # bent, but held away from the palm

#: Screen axes: x right, y down, z away from the camera.
#: Canonically the hand is palm-to-camera with the fingers pointing up.
KNUCKLES = {
    "index": (-0.06, 0.00),
    "middle": (-0.02, -0.01),
    "ring": (0.02, 0.00),
    "pinky": (0.06, 0.01),
}


def rotate(points, pivot, yaw=0.0, pitch=0.0, roll=0.0):
    """Turn the hand about the wrist.

    yaw   -- about the vertical: face-on through three-quarter to profile
    pitch -- about the horizontal: fingers up through to aimed at the lens
    roll  -- in the plane of the screen: tilting the hand over
    """

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    turned = []

    for point in points:
        x = point.x - pivot.x
        y = point.y - pivot.y
        z = point.z - pivot.z

        y, z = y * cp - z * sp, y * sp + z * cp   # pitch
        x, z = x * cy + z * sy, -x * sy + z * cy  # yaw
        x, y = x * cr - y * sr, x * sr + y * cr   # roll

        turned.append(Point(pivot.x + x, pivot.y + y, pivot.z + z))

    return turned


def make_hand(index=EXTENDED, middle=EXTENDED, ring=EXTENDED, pinky=EXTENDED,
              cx=0.50, cy=0.50, yaw=0.0, pitch=0.0, roll=0.0, pinch=False):
    """Build 21 landmarks, then turn them to the requested viewpoint."""

    wrist = Point(cx, cy + 0.18, 0.0)

    points = [wrist]

    # The thumb, which the pinch needs: out to the side of the palm when
    # the hand is open, and tucked against the index fingertip when it is
    # pinching.
    if pinch:
        thumb_tip = Point(cx - 0.06, cy - 0.048, -0.010)
    else:
        thumb_tip = Point(cx - 0.115, cy + 0.030)

    points += [Point(cx - 0.085, cy + 0.140),
               Point(cx - 0.100, cy + 0.100),
               Point(cx - 0.110, cy + 0.065),
               thumb_tip]

    states = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}

    for name in ("index", "middle", "ring", "pinky"):
        dx, dy = KNUCKLES[name]
        mx, my = cx + dx, cy + dy

        points.append(Point(mx, my, 0.0))

        if states[name] == EXTENDED:
            # Straight: knuckle to tip is the whole length of the finger.
            points.append(Point(mx, my - 0.045, 0.0))
            points.append(Point(mx, my - 0.080, 0.0))
            points.append(Point(mx, my - 0.115, 0.0))
        elif states[name] == LOOSE:
            # Bent, but with a gap between the fingers and the palm: a hand
            # resting or half-closed, which is not a gesture.
            points.append(Point(mx, my - 0.045, 0.000))
            points.append(Point(mx, my - 0.065, -0.025))
            points.append(Point(mx, my - 0.060, -0.055))

        else:
            # Shut: the fingertip comes right back to the knuckle.
            points.append(Point(mx, my - 0.040, -0.010))
            points.append(Point(mx, my - 0.050, -0.040))
            points.append(Point(mx, my - 0.005, -0.045))

    assert len(points) == 21, len(points)

    if yaw or pitch or roll:
        points = rotate(points, wrist, yaw, pitch, roll)

    return points


AT_CAMERA = math.radians(90)  # pitch that aims the fingers down the lens


def peace_sign(cx=0.50, roll=0.0):
    """Two fingers up, palm toward the camera."""

    return make_hand(EXTENDED, EXTENDED, CURLED, CURLED, cx=cx, roll=roll)


def gun(cx=0.50, turn=0.0):
    """Two fingers aimed at the camera, wrist turned by ``turn``.

    Positive turn points the fingers to the user's right.
    """

    return make_hand(EXTENDED, EXTENDED, CURLED, CURLED,
                     cx=cx, pitch=AT_CAMERA, yaw=-turn)


def fist(**viewpoint):
    return make_hand(CURLED, CURLED, CURLED, CURLED, **viewpoint)


def pointing(**viewpoint):
    return make_hand(EXTENDED, CURLED, CURLED, CURLED, **viewpoint)


HAND = "Left"  # what MediaPipe calls a right hand in a mirrored frame

#: Face-on, both three-quarter views, both profiles, and from behind.
VIEWPOINTS = [math.radians(degrees) for degrees in
              (-180, -135, -90, -45, -20, 0, 20, 45, 90, 135, 180)]

#: Anything but the back of the hand.  Side-on counts: a fist from the
#: side is still a fist.
NOT_FROM_BEHIND = [math.radians(degrees) for degrees in
                   (-90, -45, -20, 0, 20, 45, 90)]

#: Looking at the back of the hand, which is what a hand at a keyboard
#: shows the camera.
FROM_BEHIND = [math.radians(degrees) for degrees in (-180, -135, 135, 180)]


class GestureTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        motion.now = self.clock
        vision.reset_state()
        motion._state.cooldown_until = 0.0

    def tearDown(self):
        import time as real_time
        motion.now = real_time.time

    def hold_pose(self, hand, seconds=0.25, frames=6):
        """Present a still pose before moving, as a real hand does.

        A swipe now requires the pose to persist briefly before it counts,
        which is what stops a hand reaching across the desk from firing
        one.  These tests used to feed the movement with no hand shown
        beforehand, which no real gesture does.
        """

        for _ in range(frames):
            motion.detect_swipe(hand, HAND)
            self.clock.tick(seconds / frames)

    def settle(self, hand, frames=5):
        """Hold a pose long enough for the stabiliser to accept it."""

        result = "UNKNOWN"
        for _ in range(frames):
            result = gestures.detect_gesture(hand, HAND)
        return result


# --- static gestures -------------------------------------------------------

class TestStaticGestures(GestureTestCase):
    def test_open_palm(self):
        self.assertEqual(self.settle(make_hand()), "OPEN_PALM")

    def test_fist(self):
        self.assertEqual(self.settle(fist()), "FIST")

    def test_point(self):
        self.assertEqual(self.settle(pointing()), "POINT")

    def test_two_finger(self):
        self.assertEqual(self.settle(peace_sign()), "TWO_FINGER")

    def test_needs_several_frames(self):
        """One good frame is not a gesture; that is how flicker gets in."""

        self.assertEqual(gestures.detect_gesture(make_hand(), HAND), "UNKNOWN")

    def test_a_half_closed_hand_is_not_a_fist(self):
        """Reported from real use: fingers bent but clear of the palm were
        being taken as a fist, so play/pause fired at a resting hand."""

        loose = make_hand(LOOSE, LOOSE, LOOSE, LOOSE)
        self.assertEqual(self.settle(loose), "UNKNOWN")

    def test_a_punch_is_not_a_fist(self):
        """A punch shows the camera its knuckles rather than its palm.

        Deliberate: the back of a hand is what the camera sees of someone
        typing, or resting their hand on the desk, and that should ask for
        nothing.  The gesture is a fist held up, palm out.
        """

        self.assertEqual(self.settle(fist(yaw=math.radians(180))), "UNKNOWN")

    def test_pointing_at_the_camera_is_a_point(self):
        """A finger aimed at the lens is foreshortened to almost nothing on
        screen, but it is still full length in space."""

        self.assertEqual(self.settle(pointing(pitch=AT_CAMERA)), "POINT")


class TestEveryAngle(GestureTestCase):
    """The same gesture, turned every which way.

    Finger extension is judged from 3D distances, which do not change when
    the hand turns -- so none of these should depend on the viewpoint.
    """

    #: Turns that still show the palm.  A fist and an open hand are only
    #: recognised there; the pointing and two-finger poses do not care.
    PALM_PITCHES = (math.radians(-45), 0.0, math.radians(45))

    def check_all_views(self, builder, expected, yaws=None, pitches=None):
        for yaw in (VIEWPOINTS if yaws is None else yaws):
            for pitch in (pitches if pitches is not None else
                          (0.0, math.radians(45), AT_CAMERA,
                           math.radians(-45))):
                with self.subTest(yaw=round(math.degrees(yaw)),
                                  pitch=round(math.degrees(pitch))):
                    vision.reset_state()
                    self.assertEqual(
                        self.settle(builder(yaw=yaw, pitch=pitch)), expected)

    def test_fist_from_any_side_but_behind(self):
        """Side profile included: a fist from the side is still a fist."""

        self.check_all_views(fist, "FIST",
                             yaws=NOT_FROM_BEHIND, pitches=self.PALM_PITCHES)

    def test_fist_is_not_read_from_the_back_of_the_hand(self):
        """A hand resting knuckles-out is not asking for anything."""

        self.check_all_views(fist, "UNKNOWN",
                             yaws=FROM_BEHIND, pitches=self.PALM_PITCHES)

    def test_open_hand_is_not_read_from_the_back_either(self):
        self.check_all_views(make_hand, "UNKNOWN",
                             yaws=FROM_BEHIND, pitches=self.PALM_PITCHES)

    def test_point_from_every_angle(self):
        """Three-quarter view used to report this as TWO_FINGER."""

        self.check_all_views(pointing, "POINT")

    def test_open_palm_from_any_side_but_behind(self):
        self.check_all_views(make_hand, "OPEN_PALM",
                             yaws=NOT_FROM_BEHIND, pitches=self.PALM_PITCHES)

    def test_two_finger_from_every_angle(self):
        def two(**viewpoint):
            return make_hand(EXTENDED, EXTENDED, CURLED, CURLED, **viewpoint)

        self.check_all_views(two, "TWO_FINGER")

    def test_rolling_the_hand_over_changes_nothing(self):
        """Tilting the hand in the plane of the screen is still the same
        gesture, upside down included."""

        for degrees in (-90, -45, 0, 45, 90, 180):
            with self.subTest(roll=degrees):
                vision.reset_state()
                self.assertEqual(
                    self.settle(fist(roll=math.radians(degrees))), "FIST")


class TestFingerMeasure(GestureTestCase):
    """The measurement everything else is built on."""

    def test_straight_finger_scores_near_one(self):
        self.assertGreater(
            hand_state.finger_extension(make_hand(), 8, 6, 5), 0.95)

    def test_curled_finger_scores_low(self):
        self.assertLess(
            hand_state.finger_extension(fist(), 8, 6, 5), 0.7)

    def test_score_survives_rotation(self):
        """The whole point: turning the hand must not change the answer."""

        face_on = hand_state.finger_extension(make_hand(), 8, 6, 5)

        for yaw in VIEWPOINTS:
            with self.subTest(yaw=round(math.degrees(yaw))):
                turned = hand_state.finger_extension(make_hand(yaw=yaw), 8, 6, 5)
                self.assertAlmostEqual(face_on, turned, places=6)


# --- pose recognition ------------------------------------------------------

class TestTwoFingerPose(GestureTestCase):
    def test_two_fingers_aimed_at_the_camera(self):
        self.assertEqual(gestures.pose_kind(gun()), gestures.POSE_TWO_FINGER)

    def test_two_fingers_held_up(self):
        self.assertEqual(gestures.pose_kind(peace_sign()),
                         gestures.POSE_TWO_FINGER)

    def test_an_open_hand_is_a_pose_of_its_own(self):
        """It arms a movement, but not this one.

        An open hand raised and lowered is brightness; two fingers raised
        and lowered is volume.  Which pose the movement was made from is
        the whole of what separates them.
        """

        self.assertEqual(gestures.pose_kind(make_hand()),
                         gestures.POSE_OPEN_PALM)

    def test_fist_is_not_the_pose(self):
        self.assertIsNone(gestures.pose_kind(fist()))

    def test_pose_found_at_any_yaw(self):
        """The fingers decide the pose; the angle only decides which one."""

        for yaw in VIEWPOINTS:
            with self.subTest(yaw=round(math.degrees(yaw))):
                self.assertIsNotNone(gestures.pose_kind(
                    make_hand(EXTENDED, EXTENDED, CURLED, CURLED, yaw=yaw)))


# --- swipes ----------------------------------------------------------------

class TestGunSwipe(GestureTestCase):
    """Wrist held still, two fingers turned left or right."""

    def turn(self, start, end, seconds, frames=12, pose=True, cx=0.50):
        if pose:
            self.hold_pose(gun(cx=cx, turn=start))
        result = None
        for i in range(frames):
            amount = start + (end - start) * (i / (frames - 1))
            landmarks = gun(cx=cx, turn=amount) if pose else fist(cx=cx)
            result = result or motion.detect_swipe(landmarks, HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def test_swipe_right(self):
        self.assertEqual(self.turn(-0.6, 0.6, 0.30), "SWIPE_RIGHT")

    def test_swipe_left(self):
        self.assertEqual(self.turn(0.6, -0.6, 0.30), "SWIPE_LEFT")

    def test_turning_from_centre_is_enough(self):
        """You should not have to wind up first."""

        self.assertEqual(self.turn(0.0, 0.7, 0.30), "SWIPE_RIGHT")

    def test_slow_turn_is_not_a_swipe(self):
        self.assertIsNone(self.turn(-0.6, 0.6, 4.0, frames=40))

    def test_small_turn_is_not_a_swipe(self):
        self.assertIsNone(self.turn(0.0, 0.1, 0.30))

    def test_wrong_pose_is_not_a_swipe(self):
        self.assertIsNone(self.turn(-0.6, 0.6, 0.30, pose=False))

    def test_wobble_is_not_a_swipe(self):
        """A shaking hand turns a lot without turning anywhere."""

        self.hold_pose(gun(turn=0.0))
        result = None
        for i in range(14):
            result = result or motion.detect_swipe(
                gun(turn=0.15 if i % 2 else -0.15), HAND)
            self.clock.tick(0.025)
        self.assertIsNone(result)

    def test_carrying_the_gun_pose_across_is_not_a_swipe(self):
        """Aiming at the camera and reaching for the keyboard must do nothing.

        Aim is measured relative to the wrist, so moving the hand without
        turning it reads as no movement at all.
        """

        self.hold_pose(gun(cx=0.30))
        result = None
        for i in range(12):
            cx = 0.30 + 0.40 * (i / 11)
            result = result or motion.detect_swipe(gun(cx=cx, turn=0.0), HAND)
            self.clock.tick(0.30 / 11)
        self.assertIsNone(result)

    def test_a_bad_frame_does_not_lose_the_swipe(self):
        """Aim is recorded every frame, so one misclassified frame in the
        middle costs a sample, not the gesture."""

        result = None
        for i in range(12):
            amount = -0.6 + 1.2 * (i / 11)
            landmarks = make_hand() if i == 6 else gun(turn=amount)
            result = result or motion.detect_swipe(landmarks, HAND)
            self.clock.tick(0.30 / 11)
        self.assertEqual(result, "SWIPE_RIGHT")

    def test_cooldown_blocks_an_immediate_second_swipe(self):
        self.assertEqual(self.turn(-0.6, 0.6, 0.30), "SWIPE_RIGHT")

        fired_at = motion._state.cooldown_until - motion.SWIPE_COOLDOWN

        # A second gesture begun at once: the shortest one that could
        # possibly register, so the whole of it lands inside the cooldown.
        self.hold_pose(gun(turn=-0.6), seconds=0.16, frames=4)

        result = None
        for i in range(6):
            result = result or motion.detect_swipe(gun(turn=-0.6 + 1.2 * i / 5),
                                                   HAND)
            self.clock.tick(0.02)

        self.assertLess(self.clock.now, fired_at + motion.SWIPE_COOLDOWN,
                        "the test itself ran past the cooldown")
        self.assertIsNone(result)

    def test_swipe_allowed_again_after_cooldown(self):
        self.assertEqual(self.turn(-0.6, 0.6, 0.30), "SWIPE_RIGHT")
        self.clock.tick(motion.SWIPE_COOLDOWN + 0.1)
        self.assertEqual(self.turn(-0.6, 0.6, 0.30), "SWIPE_RIGHT")

    def test_position_in_frame_does_not_matter(self):
        for cx in (0.25, 0.50, 0.75):
            with self.subTest(cx=cx):
                vision.reset_state()
                motion._state.cooldown_until = 0.0
                self.assertEqual(
                    self.turn(-0.6, 0.6, 0.30, cx=cx), "SWIPE_RIGHT")

    def test_debug_state_is_populated(self):
        motion.detect_swipe(gun(), HAND)
        state = motion.debug_state()
        for key in ("pose", "armed", "aim", "turn", "slide", "speed", "agree"):
            self.assertIn(key, state)


class TestRestingHandAsksForNothing(GestureTestCase):
    """A hand doing nothing must read as nothing.

    Reported after the pose recorded during setup -- the hand at rest --
    started firing gestures.  A resting hand has fingers straighter than a
    fist and slacker than a spread one, and with a single line to fall on
    it landed on "open", which is the gesture that turns SARV off.
    """

    def resting(self, **viewpoint):
        return make_hand(LOOSE, LOOSE, LOOSE, LOOSE, **viewpoint)

    def test_a_resting_hand_is_not_an_open_hand(self):
        self.assertEqual(self.settle(self.resting()), "UNKNOWN")

    def test_a_resting_hand_is_not_a_fist_either(self):
        """The other line it has to fall between."""

        self.assertNotEqual(self.settle(self.resting()), "FIST")

    def test_a_properly_open_hand_still_counts(self):
        self.assertEqual(self.settle(make_hand()), "OPEN_PALM")

    def test_at_rest_from_several_angles(self):
        for yaw in (-45, -20, 0, 20, 45):
            with self.subTest(yaw=yaw):
                vision.reset_state()
                self.assertEqual(
                    self.settle(self.resting(yaw=math.radians(yaw))),
                    "UNKNOWN")


class TestItSaysWhy(GestureTestCase):
    """Every pose gets an explanation, and the right one.

    Written after three attempts at guessing which test was refusing a
    gesture, all of which missed.  "It does not recognise my fist" is not
    something a threshold can be adjusted from; a line saying which check
    said no, and what it measured, is.
    """

    def test_a_fist_says_so(self):
        self.assertIn("FIST", gestures.explain(fist(), HAND))

    def test_a_resting_hand_says_what_it_is_short_of(self):
        resting = make_hand(LOOSE, LOOSE, LOOSE, LOOSE)

        self.assertIn("fist", gestures.explain(resting, HAND))

    def test_a_slack_open_hand_says_which_finger_let_it_down(self):
        """Four fingers out but none of them straight enough."""

        slack = make_hand()

        for tip, pip, mcp in hand_state.FINGERS.values():
            slack[tip] = Point(slack[pip].x, slack[pip].y - 0.02,
                               slack[pip].z + 0.02)

        said = gestures.explain(slack, HAND)

        self.assertTrue("slack" in said or "fist" in said, said)

    def test_the_back_of_a_hand_says_so(self):
        said = gestures.explain(fist(yaw=math.radians(180)), HAND)

        self.assertIn("back of hand", said)

    def test_every_pose_gets_some_answer(self):
        for hand in (fist(), peace_sign(), pointing(),
                     make_hand(), make_hand(LOOSE, LOOSE, LOOSE, LOOSE)):
            self.assertTrue(gestures.explain(hand, HAND).strip())


class TestVolumeOnTwoFingers(GestureTestCase):
    """Raising and lowering the same two fingers that seek.

    Sharing one pose is what this was moved off once already: a turn
    raises the hand a little and a raise turns it a little, so every
    movement has to be told apart from the other one.  It is back by
    request, with the rule that a movement must be clearly one of them.
    """

    def peace(self, cy=0.50, roll=0.0):
        return make_hand(EXTENDED, EXTENDED, CURLED, CURLED, cy=cy, roll=roll)

    def move(self, start, end, seconds=0.30, frames=12, roll=0.0, hold=True):
        if hold:
            self.hold_pose(self.peace(cy=start, roll=roll))

        result = None
        for i in range(frames):
            cy = start + (end - start) * (i / (frames - 1))
            result = result or motion.detect_swipe(
                self.peace(cy=cy, roll=roll), HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def rest(self, cy, seconds=1.30, frames=26):
        for _ in range(frames):
            motion.detect_swipe(self.peace(cy=cy), HAND)
            self.clock.tick(seconds / frames)

    def test_raising_turns_it_up(self):
        self.assertEqual(self.move(0.55, 0.35), "SWIPE_UP")

    def test_lowering_turns_it_down(self):
        self.assertEqual(self.move(0.45, 0.65), "SWIPE_DOWN")

    def test_putting_the_hand_back_does_nothing(self):
        self.assertEqual(self.move(0.55, 0.35), "SWIPE_UP")
        self.assertIsNone(self.move(0.35, 0.55, hold=False))

    def test_lowering_after_settling_at_a_new_height(self):
        self.assertEqual(self.move(0.55, 0.35), "SWIPE_UP")
        self.rest(0.35)
        self.assertEqual(self.move(0.35, 0.55, hold=False), "SWIPE_DOWN")

    def test_a_slow_raise_is_not_a_gesture(self):
        self.assertIsNone(self.move(0.55, 0.35, seconds=4.0, frames=40))

    def test_turning_still_seeks(self):
        """The other half of the pose has to keep working."""

        result = None
        self.hold_pose(self.peace(roll=math.radians(-25)))

        for i in range(12):
            result = result or motion.detect_swipe(
                self.peace(roll=math.radians(-25 + 50 * (i / 11))), HAND)
            self.clock.tick(0.30 / 11)

        self.assertEqual(result, "SWIPE_RIGHT")

    def test_a_turn_does_not_change_the_volume(self):
        result = None
        self.hold_pose(self.peace(roll=math.radians(-25)))

        for i in range(12):
            result = result or motion.detect_swipe(
                self.peace(roll=math.radians(-25 + 50 * (i / 11))), HAND)
            self.clock.tick(0.30 / 11)

        self.assertNotIn(result, ("SWIPE_UP", "SWIPE_DOWN"))

    def test_half_one_and_half_the_other_seeks(self):
        """A movement that is both reads as the turn.

        The turn is asked on its own terms because it is the more
        distinctive measurement -- which way the fingers point, rather than
        where the hand is -- and a wrist turned sideways lifts the hand a
        little in the doing, which should not cost the gesture.  Volume is
        the half that has to be clean.
        """

        self.hold_pose(self.peace(cy=0.55, roll=math.radians(-20)))

        result = None
        for i in range(12):
            share = i / 11
            result = result or motion.detect_swipe(
                self.peace(cy=0.55 - 0.14 * share,
                           roll=math.radians(-20 + 34 * share)), HAND)
            self.clock.tick(0.30 / 11)

        self.assertNotIn(result, ("SWIPE_UP", "SWIPE_DOWN"))


class TestOneFrameCannotBeAGesture(GestureTestCase):
    """A hand cannot cross a gesture's worth of ground in a single frame.

    A misread landmark can, and it arrives looking like the best gesture
    the app has ever seen: one enormous step, so fast by any measure, and
    perfectly direct, because two points cannot disagree about a
    direction.  Speed and straightness both vouch for it.
    """

    def peace(self, roll=0.0, cy=0.55):
        return make_hand(EXTENDED, EXTENDED, CURLED, CURLED, cy=cy,
                         roll=math.radians(roll))

    def play(self, rolls=None, heights=None, seconds=0.30):
        count = len(rolls or heights)

        for _ in range(20):
            motion.detect_swipe(
                self.peace((rolls or [0.0])[0], (heights or [0.55])[0]), HAND)
            self.clock.tick(0.045)

        fired = None

        for i in range(count):
            fired = fired or motion.detect_swipe(
                self.peace(rolls[i] if rolls else 0.0,
                           heights[i] if heights else 0.55), HAND)
            self.clock.tick(seconds / max(1, count - 1))

        return fired

    def test_a_single_bad_frame_sideways_fires_nothing(self):
        self.assertIsNone(
            self.play(rolls=[-25, -25, 25, -25, -25, -25], seconds=0.20))

    def test_a_single_bad_frame_vertically_fires_nothing(self):
        self.assertIsNone(
            self.play(heights=[0.55, 0.55, 0.30, 0.55, 0.55, 0.55],
                      seconds=0.20))

    def test_a_waggle_that_happens_to_end_turned_fires_nothing(self):
        """Its first leg was one frame wide, and that leg was what fired."""

        self.assertIsNone(
            self.play(rolls=[-25, 10, -20, 12, -18, 15, 22], seconds=0.40))

    def test_a_quick_flick_still_counts(self):
        """The point is to refuse one frame, not to refuse speed.

        Six frames in a sixth of a second is about as fast as a wrist
        goes, and it has to keep working.
        """

        rolls = [-25 + 50 * i / 5 for i in range(6)]

        self.assertEqual(self.play(rolls=rolls, seconds=0.16), "SWIPE_RIGHT")

    def test_an_ordinary_turn_still_counts(self):
        rolls = [-25 + 50 * i / 13 for i in range(14)]

        self.assertEqual(self.play(rolls=rolls), "SWIPE_RIGHT")

    def test_an_ordinary_raise_still_counts(self):
        heights = [0.55 - 0.20 * i / 13 for i in range(14)]

        self.assertEqual(self.play(heights=heights, seconds=0.35), "SWIPE_UP")


class TestTheFourDirectionsAreDistinct(GestureTestCase):
    """Up, down, left and right, told apart on the same pose.

    Turning the wrist raises the hand a little and raising it turns the
    wrist a little, so every movement has to say which of the two it was.
    Both are asked the same question -- is the other one a small share of
    me -- and a movement that is half of each is a diagonal nobody meant,
    which fires nothing rather than whichever happened to be asked first.
    """

    def peace(self, cy=0.55, roll=0.0):
        return make_hand(EXTENDED, EXTENDED, CURLED, CURLED, cy=cy,
                         roll=math.radians(roll))

    def move(self, roll_by=0.0, rise_by=0.0, frames=14, seconds=0.30):
        for _ in range(20):
            motion.detect_swipe(self.peace(), HAND)
            self.clock.tick(0.045)

        fired = None

        for i in range(frames):
            share = i / (frames - 1)
            fired = fired or motion.detect_swipe(
                self.peace(cy=0.55 - rise_by * share,
                           roll=roll_by * share), HAND)
            self.clock.tick(seconds / (frames - 1))

        return fired

    def test_up(self):
        self.assertEqual(self.move(rise_by=0.20), "SWIPE_UP")

    def test_down(self):
        self.assertEqual(self.move(rise_by=-0.20), "SWIPE_DOWN")

    def test_right(self):
        self.assertEqual(self.move(roll_by=50), "SWIPE_RIGHT")

    def test_left(self):
        self.assertEqual(self.move(roll_by=-50), "SWIPE_LEFT")

    def test_mostly_up_is_up_despite_a_turn(self):
        """A raise turns the wrist a little; that must not cost it."""

        self.assertEqual(self.move(roll_by=12, rise_by=0.15), "SWIPE_UP")

    def test_mostly_sideways_is_sideways_despite_a_rise(self):
        """And a turn raises the hand a little.

        This is the one that was got wrong twice: first by never asking,
        which made every diagonal a seek, then by asking against a
        guessed size, which stopped turns registering at all.
        """

        self.assertEqual(self.move(roll_by=38, rise_by=0.05), "SWIPE_RIGHT")

    def test_half_of_each_fires_nothing(self):
        """Twenty degrees was used here and sat exactly on the boundary,
        which is no way to test one.  Twenty-five is plainly a diagonal."""

        for roll, rise in ((25, 0.12), (-25, 0.12), (25, -0.12)):
            with self.subTest(turn=roll, rise=rise):
                self.setUp()

                self.assertIsNone(self.move(roll_by=roll, rise_by=rise))

    def test_a_drop_may_roll_the_wrist_without_losing_it(self):
        """An arm coming down rotates the wrist, and that is not a
        gesture -- it is what an arm does.

        Reported as an open palm lowered being very hard to register.  A
        turn through fifty degrees barely moves the hand up or down, so
        the two were never going to want the same allowance, and the one
        they shared refused an ordinary lower.
        """

        for roll in (0, 5, 10):
            with self.subTest(rolling=roll):
                self.setUp()

                self.assertEqual(self.move(roll_by=roll, rise_by=-0.20),
                                 "SWIPE_DOWN")

    def test_a_turn_is_still_asked_the_stricter_question(self):
        """It has no such excuse: a wrist turning does not raise a hand."""

        self.assertIsNone(self.move(roll_by=25, rise_by=0.12))

    def test_a_lost_frame_cannot_move_the_volume(self):
        """The resting height is taken from several frames, not one.

        Losing the pose resets that height, and a frame bad enough to
        lose the pose is a frame whose position is not to be trusted --
        so taken from one sample, the bad frame became the height every
        later raise was judged against.
        """

        fired = None

        for i in range(12):
            hand = make_hand() if i == 6 else self.peace(roll=-30 + 60 * i / 11)
            fired = fired or motion.detect_swipe(hand, HAND)
            self.clock.tick(0.30 / 11)

        self.assertNotIn(fired, ("SWIPE_UP", "SWIPE_DOWN"))


class TestAFistIsAFistHoweverItIsMade(GestureTestCase):
    """Two measurements, either of which closes a finger.

    Reported as the fist not being recognised from the side.  It was not
    being recognised from anywhere: the calibration had been done with a
    hard clench, which put the reach line at 0.99, and an ordinary fist
    reaches 1.05.  The side view was never the difference -- reach reads
    the same at every angle, which is what it was built to do.

    What separates a hard fist from a casual one is exactly what reach
    measures: how far the fingertips end up from the wrist.  How curled
    each finger is barely moves between them, so it is asked as well.
    """

    def fist(self, **view):
        return make_hand(CURLED, CURLED, CURLED, CURLED, **view)

    def tight(self):
        """The reach line a hard-clenched calibration produces."""

        hand_state.FIST_REACH = 0.99

    def setUp(self):
        super().setUp()
        self.was = (hand_state.FIST_REACH, hand_state.FIST_CURL)

    def tearDown(self):
        hand_state.FIST_REACH, hand_state.FIST_CURL = self.was
        super().tearDown()

    def test_an_ordinary_fist_against_a_hard_clenched_line(self):
        """The reported failure, and the case that matters."""

        self.tight()

        self.assertEqual(self.settle(self.fist()), "FIST")

    def test_from_every_angle_but_behind(self):
        self.tight()

        for degrees in (0, 20, 45, 70, 90, -45, -90):
            with self.subTest(turned=degrees):
                self.assertEqual(
                    self.settle(self.fist(yaw=math.radians(degrees))), "FIST")

    def test_the_back_of_a_hand_is_still_not_a_fist(self):
        """What the camera sees of someone resting a hand on a desk."""

        self.tight()

        self.assertNotEqual(
            self.settle(self.fist(yaw=math.radians(180))), "FIST")

    def test_a_resting_hand_is_still_not_a_fist(self):
        """The curl line has to stay above a fist and below a slack hand,
        or this buys a false play/pause every time a hand lies still."""

        self.tight()

        self.assertNotEqual(
            self.settle(make_hand(LOOSE, LOOSE, LOOSE, LOOSE)), "FIST")

    def test_nor_are_the_poses_that_hold_fingers_out(self):
        self.tight()

        for name, fingers in (
            ("two fingers", (EXTENDED, EXTENDED, CURLED, CURLED)),
            ("point", (EXTENDED, CURLED, CURLED, CURLED)),
            ("open palm", (EXTENDED, EXTENDED, EXTENDED, EXTENDED)),
        ):
            with self.subTest(pose=name):
                self.assertNotEqual(self.settle(make_hand(*fingers)), "FIST")

    def test_reach_alone_still_finds_a_tight_fist(self):
        """The older route has not been replaced, only joined."""

        hand_state.FIST_CURL = 0.0        # curl can answer for nothing

        self.assertEqual(self.settle(self.fist()), "FIST")


class TestEveryPoseAtADistance(GestureTestCase):
    """The same hands, shrunk, with landmark error added.

    Error is roughly fixed in pixels, so it grows against the hand as the
    hand shrinks -- which is the whole of why range is hard.  These hands
    are geometrically perfect and a real one seen small is also blurred
    and partly lost, so this is the optimistic case; what it is good for
    is catching a threshold that is secretly a size rather than a ratio.
    """

    #: Two pixels of error in a 1280-wide frame.
    BLUR = 2.0 / 1280

    POSES = {
        "FIST": (CURLED, CURLED, CURLED, CURLED),
        "POINT": (EXTENDED, CURLED, CURLED, CURLED),
        "TWO_FINGER": (EXTENDED, EXTENDED, CURLED, CURLED),
        "OPEN_PALM": (EXTENDED, EXTENDED, EXTENDED, EXTENDED),
    }

    def resized(self, points, wanted):
        wrist = points[0]
        scale = wanted / hand_state.hand_scale(points)

        return [Point(wrist.x + (point.x - wrist.x) * scale,
                      wrist.y + (point.y - wrist.y) * scale,
                      wrist.z + (point.z - wrist.z) * scale)
                for point in points]

    def blurred(self, points, rng):
        return [Point(point.x + rng.gauss(0, self.BLUR),
                      point.y + rng.gauss(0, self.BLUR),
                      point.z + rng.gauss(0, self.BLUR))
                for point in points]

    def accuracy(self, wanted, scale, trials=40, seed=3):
        rng = random.Random(seed)
        base = self.resized(make_hand(*self.POSES[wanted]), scale)
        right = 0

        for _ in range(trials):
            vision.reset_state()

            for _ in range(6):
                got = gestures.detect_gesture(self.blurred(base, rng), HAND)

            right += got == wanted

        return right / trials

    def test_every_pose_holds_up_close(self):
        for wanted in self.POSES:
            with self.subTest(pose=wanted):
                self.assertGreater(self.accuracy(wanted, 0.12), 0.95)

    def test_pointing_and_two_fingers_reach_across_a_room(self):
        """A hand a twentieth of the frame: a metre and a half or so on a
        laptop webcam.  Two fingers carries four of the seven commands,
        so this is the one that matters most."""

        for wanted in ("POINT", "TWO_FINGER"):
            with self.subTest(pose=wanted):
                self.assertGreater(self.accuracy(wanted, 0.05), 0.90)

    def test_two_fingers_reaches_furthest(self):
        self.assertGreater(self.accuracy("TWO_FINGER", 0.035), 0.90)

    def test_an_open_hand_is_the_shortest_ranged_of_them(self):
        """Worth stating, because brightness rides on it.

        It asks the most of the reading: every finger has to be not
        merely out but properly straight, and one finger misread by
        enough is all it takes.  That strictness is not removable -- a
        hand at rest is told from an open one by the pinky alone, on the
        calibration that prompted this -- so an open hand simply has to
        be nearer.
        """

        self.assertGreater(self.accuracy("OPEN_PALM", 0.08), 0.95)
        self.assertLess(self.accuracy("OPEN_PALM", 0.03), 0.50)

    def test_nothing_is_secretly_measured_in_pixels(self):
        """A threshold that is a size rather than a ratio shows up here as
        a pose that works at one distance and not another with no noise
        involved at all."""

        for wanted in self.POSES:
            for scale in (0.20, 0.10, 0.05, 0.03):
                with self.subTest(pose=wanted, scale=scale):
                    vision.reset_state()
                    hand = self.resized(make_hand(*self.POSES[wanted]), scale)

                    for _ in range(6):
                        got = gestures.detect_gesture(hand, HAND)

                    self.assertEqual(got, wanted)


class TestBrightnessOnAnOpenPalm(GestureTestCase):
    """An open hand raised and lowered, which is brightness.

    It needed no recording of its own: the pose was already measured and
    so was the movement, which is what measuring the two apart was for.
    What separates it from volume is only which hand made the movement.
    """

    TWO = (EXTENDED, EXTENDED, CURLED, CURLED)
    PALM = (EXTENDED, EXTENDED, EXTENDED, EXTENDED)

    def hand(self, fingers, cy=0.55, roll=0.0, **view):
        return make_hand(*fingers, cy=cy, roll=math.radians(roll), **view)

    def move(self, fingers, rise=0.0, roll=0.0, frames=14, seconds=0.30,
             **view):
        for _ in range(20):
            motion.detect_swipe(self.hand(fingers, **view), HAND)
            self.clock.tick(0.045)

        fired = None

        for i in range(frames):
            share = i / (frames - 1)
            fired = fired or motion.detect_swipe(
                self.hand(fingers, cy=0.55 - rise * share,
                          roll=roll * share, **view), HAND)
            self.clock.tick(seconds / (frames - 1))

        return fired

    def test_raising_an_open_hand(self):
        self.assertEqual(self.move(self.PALM, rise=0.20), "PALM_UP")

    def test_lowering_an_open_hand(self):
        self.assertEqual(self.move(self.PALM, rise=-0.20), "PALM_DOWN")

    def test_the_same_movement_on_two_fingers_is_still_volume(self):
        self.assertEqual(self.move(self.TWO, rise=0.20), "SWIPE_UP")
        self.setUp()
        self.assertEqual(self.move(self.TWO, rise=-0.20), "SWIPE_DOWN")

    def test_turning_an_open_hand_does_nothing(self):
        """Seeking stays on two fingers.  An open hand turning is a hand
        being turned over, not an instruction."""

        self.assertIsNone(self.move(self.PALM, roll=50))
        self.setUp()
        self.assertIsNone(self.move(self.PALM, roll=-50))

    def test_the_back_of_a_hand_does_not_arm_it(self):
        """Which is what the camera sees of someone reaching for
        something -- and reaching moves the hand, which is the whole of
        what this pose is then asked about."""

        back = {"yaw": math.radians(180)}

        self.assertIsNone(gestures.pose_kind(self.hand(self.PALM, **back)))
        self.assertIsNone(self.move(self.PALM, rise=0.20, **back))

    def test_a_resting_hand_does_not_arm_it(self):
        resting = (LOOSE, LOOSE, LOOSE, LOOSE)

        self.assertIsNone(gestures.pose_kind(self.hand(resting)))
        self.assertIsNone(self.move(resting, rise=0.20))

    def test_it_reaches_the_brightness_commands(self):
        from control import Command
        from integration.bridge import GestureRouter

        router = GestureRouter()

        self.assertEqual(router.update(None, "PALM_UP", now=1000.0),
                         Command.BRIGHTNESS_UP)
        self.assertEqual(router.update(None, "PALM_DOWN", now=1002.0),
                         Command.BRIGHTNESS_DOWN)


class TestBringingTheWristBack(GestureTestCase):
    """Reported: putting the wrist back counted as a second gesture.

    Turning back is the same movement as turning the other way, and no
    measurement of the movement itself separates them -- same size, same
    speed, same steadiness, opposite sign.  What separates them is where
    the wrist ends up: a deliberate turn leaves it somewhere new, and a
    return puts it back where it began.
    """

    def restart(self):
        """A fresh hand and a fresh clock, for looping over cases."""

        self.setUp()

    def peace(self, roll):
        return make_hand(EXTENDED, EXTENDED, CURLED, CURLED,
                         roll=math.radians(roll))

    def sweep(self, start, end, seconds, frames=14):
        """Turn the wrist from one angle to another, collecting what fires."""

        fired = []

        for i in range(frames):
            roll = start + (end - start) * (i / (frames - 1))
            got = motion.detect_swipe(self.peace(roll), HAND)

            if got:
                fired.append(got)

            self.clock.tick(seconds / (frames - 1))

        return fired

    def there_and_back(self, out, back, pause):
        """Hold the pose, turn, pause, and put the wrist back."""

        fired = self.sweep(out, out, 0.9, 20)       # holding it, to arm
        fired += self.sweep(out, back, 0.30)        # the turn
        fired += self.sweep(back, back, pause, max(3, int(pause * 30)))
        fired += self.sweep(back, out, 0.35)        # putting it back
        fired += self.sweep(out, out, 0.8, 24)      # at rest again

        return fired

    def test_the_return_counts_for_nothing_one_way(self):
        for pause in (0.2, 0.5, 1.0, 2.0, 4.0):
            with self.subTest(paused_for=pause):
                self.restart()

                self.assertEqual(self.there_and_back(-25, 25, pause),
                                 ["SWIPE_RIGHT"])

    def test_the_return_counts_for_nothing_the_other_way(self):
        for pause in (0.2, 0.5, 1.0, 2.0, 4.0):
            with self.subTest(paused_for=pause):
                self.restart()

                self.assertEqual(self.there_and_back(25, -25, pause),
                                 ["SWIPE_LEFT"])

    def test_a_long_pause_does_not_make_the_turned_angle_the_resting_one(self):
        """Where a gesture left the wrist is the one place it is not resting.

        Wherever the wrist is held becomes the angle turns are judged
        against, which it has to be -- there is no neutral to assume.  But
        a wrist paused mid-gesture has not chosen to be there, and adopting
        that angle is what made the way back a gesture of its own.
        """

        self.assertEqual(self.there_and_back(-25, 25, 4.0), ["SWIPE_RIGHT"])

    def test_turning_again_still_counts(self):
        """The point is to lose the return stroke, not the next gesture."""

        fired = []

        for _ in range(3):
            fired += self.sweep(-25, -25, 0.9, 20)
            fired += self.sweep(-25, 25, 0.30)
            fired += self.sweep(25, 25, 0.4, 12)
            fired += self.sweep(25, -25, 0.35)
            fired += self.sweep(-25, -25, 0.8, 24)

        self.assertEqual(fired, ["SWIPE_RIGHT"] * 3)

    def test_turning_straight_back_out_again_counts(self):
        """Seeking twice in a row, with no pause worth the name.

        The first attempt at this waited for the hand to go still before
        anything counted again, which loses nothing when there is a pause
        and loses the whole of the next gesture when there is not.
        """

        fired = self.sweep(-25, -25, 0.9, 20)

        for _ in range(3):
            fired += self.sweep(-25, 25, 0.30)
            fired += self.sweep(25, 25, 0.2, 6)
            fired += self.sweep(25, -25, 0.35)
            fired += self.sweep(-25, -25, 0.05, 3)

        self.assertEqual(fired, ["SWIPE_RIGHT"] * 3)


class TestTwoFingersSurvivesAMisread(GestureTestCase):
    """The folded fingers are the ones the camera reads worst.

    Held up, the ring and pinky are folded down behind the raised two,
    and a folded finger behind a hand is half guessed at.  The pose used
    to rest on both of those readings being right at once, which is the
    two readings least worth resting on.
    """

    def two(self, ring=CURLED, pinky=CURLED):
        return make_hand(EXTENDED, EXTENDED, ring, pinky)

    def test_the_plain_pose(self):
        self.assertEqual(gestures.classify(self.two(), HAND), "TWO_FINGER")
        self.assertIsNotNone(gestures.pose_kind(self.two()))

    def test_a_folded_finger_misread_for_a_moment_does_not_lose_it(self):
        """The pose is held; the bad reading is not.

        A folded ring finger reported as a straight one, for a frame or
        two, which is what a camera does with a finger hidden behind a
        hand.  It reads exactly like a third finger going up -- the only
        thing telling them apart is that this one stops.
        """

        for _ in range(6):
            gestures.pose_kind(self.two())

        for _ in range(2):
            self.assertIsNotNone(gestures.pose_kind(self.two(ring=EXTENDED)))

        self.assertIsNotNone(gestures.pose_kind(self.two()))

    def test_a_finger_that_stays_out_is_believed(self):
        """And this is the same reading that does not stop: three fingers.

        Kept apart from two, so it stays available to mean something of
        its own.
        """

        for _ in range(6):
            gestures.pose_kind(self.two())

        for _ in range(4):
            gestures.pose_kind(self.two(ring=EXTENDED))

        self.assertIsNone(gestures.pose_kind(self.two(ring=EXTENDED)))

    def test_both_being_out_is_a_different_hand(self):
        """Where the line still is.  Four fingers out is an open hand,
        which is its own pose and not this one."""

        hand = make_hand(EXTENDED, EXTENDED, EXTENDED, EXTENDED)

        self.assertEqual(gestures.classify(hand, HAND), "OPEN_PALM")
        self.assertEqual(gestures.pose_kind(hand), gestures.POSE_OPEN_PALM)

    def test_three_fingers_is_not_this_pose(self):
        """Kept apart deliberately, so it stays available to mean
        something of its own."""

        hand = make_hand(EXTENDED, EXTENDED, EXTENDED, CURLED)

        self.assertNotEqual(gestures.classify(hand, HAND), "TWO_FINGER")
        self.assertIsNone(gestures.pose_kind(hand))

    def test_a_resting_hand_is_still_not_the_pose(self):
        hand = make_hand(LOOSE, LOOSE, LOOSE, LOOSE)

        self.assertIsNone(gestures.pose_kind(hand))


class TestAFingerOnTheLine(unittest.TestCase):
    """A finger measured near the threshold, frame after frame.

    Reported as pointing being recognised only sometimes.  Every pose here
    is a statement about which fingers are out, so one finger crossing the
    line and back is a pose alternating with another one -- and pointing
    needs three fingers to stay down at once, which made it the worst
    affected.
    """

    def setUp(self):
        from vision.state_machine import FingerMemory

        self.memory = FingerMemory(band=0.07)

    def out(self, span, threshold=0.82):
        return self.memory.update({"index": span}, threshold)["index"]

    def test_with_nothing_remembered_it_simply_asks(self):
        self.assertTrue(self.out(0.90))

    def test_a_finger_clearly_down_reads_as_down(self):
        self.assertFalse(self.out(0.40))

    def test_wobble_around_the_line_does_not_flip_it(self):
        """The reading moves; the hand does not."""

        self.out(0.78)                       # settles as down

        for span in (0.80, 0.79, 0.81, 0.78, 0.80):
            self.assertFalse(self.out(span), span)

    def test_a_finger_that_is_out_stays_out_through_a_dip(self):
        self.out(0.95)

        for span in (0.84, 0.80, 0.78, 0.81):
            self.assertTrue(self.out(span), span)

    def test_a_real_fold_registers_within_three_frames(self):
        """Only noise is refused, not the movement.

        Three frames rather than one is the price of not letting a couple
        of readings decide anything: at thirty a second it is a tenth of
        a second, on a pose that is being held anyway, and it buys the
        difference between three fingers and two.
        """

        for _ in range(5):
            self.out(0.95)

        self.out(0.60)
        self.out(0.60)

        self.assertFalse(self.out(0.60))

    def test_a_real_straightening_registers_within_three_frames(self):
        for _ in range(5):
            self.out(0.40)

        self.out(0.95)
        self.out(0.95)

        self.assertTrue(self.out(0.95))

    def test_one_bad_reading_changes_nothing(self):
        """A folded finger is half hidden, and half of what is reported
        for it is a guess.  Occasionally the guess is not close.

        The band does not help here: a reading well past the line walks
        straight through a margin meant for wobble.  Only refusing to let
        one frame decide does.
        """

        for _ in range(4):
            self.out(0.58)

        self.assertFalse(self.out(0.88), "one bad frame broke the pose")
        self.assertFalse(self.out(0.58))

    def test_two_bad_readings_in_a_row_still_decide_nothing(self):
        """Which is what keeps three fingers distinct from two."""

        for _ in range(5):
            self.out(0.58)

        self.assertFalse(self.out(0.88))
        self.assertFalse(self.out(0.88))

    def test_three_in_a_row_are_believed(self):
        """Where the line is drawn, and it is drawn deliberately.

        Nothing can tell a persistent misread from a finger that really
        moved, because they are the same reading.  Somewhere it has to
        stop waiting, and a tenth of a second is where.
        """

        for _ in range(5):
            self.out(0.58)

        self.out(0.88)
        self.out(0.88)

        self.assertTrue(self.out(0.88))

    def test_forgetting_starts_it_over(self):
        self.out(0.95)
        self.memory.clear()

        self.assertFalse(self.out(0.80))


if __name__ == "__main__":
    unittest.main(verbosity=2)
