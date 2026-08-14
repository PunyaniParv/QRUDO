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
CURLED = "curled"

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
              cx=0.50, cy=0.50, yaw=0.0, pitch=0.0, roll=0.0):
    """Build 21 landmarks, then turn them to the requested viewpoint."""

    wrist = Point(cx, cy + 0.18, 0.0)

    points = [wrist]
    points += [Point(cx - 0.10, cy + 0.12)] * 4  # thumb, unused here

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
        else:
            # Curled in toward the palm, so the tip ends up near the knuckle.
            points.append(Point(mx, my - 0.040, -0.010))
            points.append(Point(mx, my - 0.050, -0.040))
            points.append(Point(mx, my - 0.020, -0.055))

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


class GestureTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        motion.now = self.clock
        vision.reset_state()
        motion._state.cooldown_until = 0.0

    def tearDown(self):
        import time as real_time
        motion.now = real_time.time

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

    def test_punch_is_a_fist(self):
        """A punch shows the camera its knuckles, not its palm."""

        self.assertEqual(self.settle(fist(yaw=math.radians(180))), "FIST")

    def test_pointing_at_the_camera_is_a_point(self):
        """A finger aimed at the lens is foreshortened to almost nothing on
        screen, but it is still full length in space."""

        self.assertEqual(self.settle(pointing(pitch=AT_CAMERA)), "POINT")


class TestEveryAngle(GestureTestCase):
    """The same gesture, turned every which way.

    Finger extension is judged from 3D distances, which do not change when
    the hand turns -- so none of these should depend on the viewpoint.
    """

    def check_all_views(self, builder, expected):
        for yaw in VIEWPOINTS:
            for pitch in (0.0, math.radians(45), AT_CAMERA, math.radians(-45)):
                with self.subTest(yaw=round(math.degrees(yaw)),
                                  pitch=round(math.degrees(pitch))):
                    vision.reset_state()
                    self.assertEqual(
                        self.settle(builder(yaw=yaw, pitch=pitch)), expected)

    def test_fist_from_every_angle(self):
        """Side profile used to miss this entirely."""

        self.check_all_views(fist, "FIST")

    def test_point_from_every_angle(self):
        """Three-quarter view used to report this as TWO_FINGER."""

        self.check_all_views(pointing, "POINT")

    def test_open_palm_from_every_angle(self):
        self.check_all_views(make_hand, "OPEN_PALM")

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
    def test_gun_pose(self):
        self.assertEqual(gestures.two_finger_pose_kind(gun()), gestures.POSE_GUN)

    def test_peace_pose(self):
        self.assertEqual(gestures.two_finger_pose_kind(peace_sign()), gestures.POSE_PEACE)

    def test_open_hand_is_not_the_pose(self):
        self.assertIsNone(gestures.two_finger_pose_kind(make_hand()))

    def test_fist_is_not_the_pose(self):
        self.assertIsNone(gestures.two_finger_pose_kind(fist()))

    def test_pose_found_at_any_yaw(self):
        """The fingers decide the pose; the angle only decides which one."""

        for yaw in VIEWPOINTS:
            with self.subTest(yaw=round(math.degrees(yaw))):
                self.assertIsNotNone(gestures.two_finger_pose_kind(
                    make_hand(EXTENDED, EXTENDED, CURLED, CURLED, yaw=yaw)))


# --- swipes ----------------------------------------------------------------

class TestGunSwipe(GestureTestCase):
    """Wrist held still, two fingers turned left or right."""

    def turn(self, start, end, seconds, frames=12, pose=True, cx=0.50):
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

        result = None
        for i in range(14):
            result = result or motion.detect_swipe(
                gun(turn=0.5 if i % 2 else -0.5), HAND)
            self.clock.tick(0.025)
        self.assertIsNone(result)

    def test_carrying_the_gun_pose_across_is_not_a_swipe(self):
        """Aiming at the camera and reaching for the keyboard must do nothing.

        Aim is measured relative to the wrist, so moving the hand without
        turning it reads as no movement at all.
        """

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
        self.assertIsNone(self.turn(-0.6, 0.6, 0.30))

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


class TestPeaceSignSwipe(GestureTestCase):
    """Two fingers up, palm to the camera: slid or tilted across."""

    def slide(self, start, end, seconds, frames=12):
        result = None
        for i in range(frames):
            cx = start + (end - start) * (i / (frames - 1))
            result = result or motion.detect_swipe(peace_sign(cx=cx), HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def test_slide_right(self):
        self.assertEqual(self.slide(0.40, 0.58, 0.30), "SWIPE_RIGHT")

    def test_slide_left(self):
        self.assertEqual(self.slide(0.58, 0.40, 0.30), "SWIPE_LEFT")

    def test_slow_slide_is_not_a_swipe(self):
        self.assertIsNone(self.slide(0.40, 0.58, 4.0, frames=40))

    def test_small_slide_is_not_a_swipe(self):
        self.assertIsNone(self.slide(0.50, 0.52, 0.30))

    def test_tilting_also_works(self):
        """Leaning the fingers over turns the aim, same as the gun pose."""

        result = None
        for i in range(12):
            roll = math.radians(-40 + 80 * (i / 11))
            result = result or motion.detect_swipe(peace_sign(roll=roll), HAND)
            self.clock.tick(0.30 / 11)
        self.assertIn(result, ("SWIPE_LEFT", "SWIPE_RIGHT"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
