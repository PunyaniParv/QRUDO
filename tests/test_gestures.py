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


def pinching(cx=0.50, cy=0.50, **viewpoint):
    """Thumb and index finger together, the rest of the hand as it falls."""

    return make_hand(LOOSE, LOOSE, LOOSE, LOOSE,
                     cx=cx, cy=cy, pinch=True, **viewpoint)


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


class TestPeaceSignSwipe(GestureTestCase):
    """Two fingers up, palm to the camera: slid or tilted across.

    Sliding is off by default -- shifting your hand a little is not a
    gesture -- so these turn it on.
    """

    def setUp(self):
        super().setUp()
        self._slide_was = motion.SWIPE_ALLOW_SLIDE
        motion.SWIPE_ALLOW_SLIDE = True

    def tearDown(self):
        motion.SWIPE_ALLOW_SLIDE = self._slide_was
        super().tearDown()

    def slide(self, start, end, seconds, frames=12):
        self.hold_pose(peace_sign(cx=start))
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

        self.hold_pose(peace_sign(roll=math.radians(-40)))
        result = None
        for i in range(12):
            roll = math.radians(-40 + 80 * (i / 11))
            result = result or motion.detect_swipe(peace_sign(roll=roll), HAND)
            self.clock.tick(0.30 / 11)
        self.assertIn(result, ("SWIPE_LEFT", "SWIPE_RIGHT"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRotatedPeaceSign(GestureTestCase):
    """The chosen gesture: two fingers facing the camera, wrist rotated.

    Everything stays in the plane of the image, so nothing here leans on
    MediaPipe's depth -- which is what made the gun pose unreliable.
    """

    def roll(self, start, end, seconds, frames=12):
        self.hold_pose(peace_sign(roll=math.radians(start)))
        result = None
        for i in range(frames):
            degrees = start + (end - start) * (i / (frames - 1))
            hand = peace_sign(roll=math.radians(degrees))
            result = result or motion.detect_swipe(hand, HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def test_rotating_right(self):
        self.assertEqual(self.roll(-30, 30, 0.30), "SWIPE_RIGHT")

    def test_rotating_left(self):
        self.assertEqual(self.roll(30, -30, 0.30), "SWIPE_LEFT")

    def test_small_rotation_is_enough(self):
        """A comfortable flick of the wrist, not a full sweep."""

        self.assertEqual(self.roll(-15, 15, 0.25), "SWIPE_RIGHT")

    def test_from_upright(self):
        """Starting straight up, without winding back first."""

        self.assertEqual(self.roll(0, 30, 0.25), "SWIPE_RIGHT")

    def test_slow_rotation_is_not_a_swipe(self):
        """Lowering your hand must not seek the video."""

        self.assertIsNone(self.roll(-30, 30, 4.0, frames=40))

    def test_pose_holds_throughout_the_rotation(self):
        for degrees in range(-60, 61, 15):
            with self.subTest(roll=degrees):
                hand = peace_sign(roll=math.radians(degrees))
                self.assertEqual(gestures.two_finger_pose_kind(hand),
                                 gestures.POSE_PEACE)


class TestFastSwipes(GestureTestCase):
    """A quick flick, which is where this used to fall apart.

    Moving fast blurs the frame, and a blurred hand is one MediaPipe
    misses -- so a fast gesture is exactly the one most likely to arrive
    with frames missing from the middle of it.
    """

    def flick(self, seconds, frames, dropped=()):
        """Rotate through 50 degrees, losing the listed frames entirely."""

        self.hold_pose(peace_sign(roll=math.radians(-25)))
        result = None
        for i in range(frames):
            degrees = -25 + 50 * (i / (frames - 1))
            if i not in dropped:
                hand = peace_sign(roll=math.radians(degrees))
                result = result or motion.detect_swipe(hand, HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def test_quick_flick(self):
        """Six frames, a fifth of a second."""

        self.assertEqual(self.flick(0.20, 6), "SWIPE_RIGHT")

    def test_very_quick_flick(self):
        """Four frames is about as short as a real gesture gets."""

        self.assertEqual(self.flick(0.13, 4), "SWIPE_RIGHT")

    def test_survives_a_lost_frame(self):
        self.assertEqual(self.flick(0.20, 8, dropped={3}), "SWIPE_RIGHT")

    def test_survives_two_lost_frames(self):
        self.assertEqual(self.flick(0.20, 8, dropped={3, 4}), "SWIPE_RIGHT")


class TestPresence(GestureTestCase):
    """How long the hand has to be gone before the gesture is abandoned."""

    def test_a_blink_is_not_a_departure(self):
        presence = gd_presence(grace=0.35)
        presence.seen(1000.0)
        self.assertFalse(presence.missing(1000.1))

    def test_a_real_departure_is(self):
        presence = gd_presence(grace=0.35)
        presence.seen(1000.0)
        self.assertTrue(presence.missing(1000.5))

    def test_reported_once_not_every_frame(self):
        presence = gd_presence(grace=0.35)
        presence.seen(1000.0)
        self.assertTrue(presence.missing(1000.5))
        self.assertFalse(presence.missing(1000.6))

    def test_nothing_to_forget_before_a_hand_appears(self):
        self.assertFalse(gd_presence().missing(1000.0))


def gd_presence(grace=0.35):
    from vision.state_machine import Presence
    return Presence(grace)


class TestReturnStroke(GestureTestCase):
    """Bringing the hand back must not count as the opposite swipe.

    Reported from real use: turn left to rewind, straighten up, and the
    straightening registered as a swipe right.  Every swipe is followed by
    a return, and a return from the left is a movement to the right --
    exactly what a deliberate rightward turn looks like.
    """

    def rotate_through(self, start, end, seconds=0.30, frames=12):
        result = None
        for i in range(frames):
            degrees = start + (end - start) * (i / (frames - 1))
            result = result or motion.detect_swipe(
                peace_sign(roll=math.radians(degrees)), HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def hold_still(self, degrees=0, seconds=0.8, frames=16):
        result = None
        for _ in range(frames):
            result = result or motion.detect_swipe(
                peace_sign(roll=math.radians(degrees)), HAND)
            self.clock.tick(seconds / frames)
        return result

    def test_returning_from_left_is_not_a_right_swipe(self):
        self.hold_pose(peace_sign())
        self.assertEqual(self.rotate_through(0, -30), "SWIPE_LEFT")
        self.assertIsNone(self.rotate_through(-30, 0),
                          "the hand coming back counted as a swipe")

    def test_returning_from_right_is_not_a_left_swipe(self):
        self.hold_pose(peace_sign())
        self.assertEqual(self.rotate_through(0, 30), "SWIPE_RIGHT")
        self.assertIsNone(self.rotate_through(30, 0),
                          "the hand coming back counted as a swipe")

    def test_a_second_swipe_still_works_after_settling(self):
        """Blocking the return must not block the next real gesture."""

        self.hold_pose(peace_sign())
        self.assertEqual(self.rotate_through(0, -30), "SWIPE_LEFT")
        self.rotate_through(-30, 0)
        self.hold_still(0)
        self.assertEqual(self.rotate_through(0, -30), "SWIPE_LEFT")

    def test_swiping_the_other_way_still_works(self):
        self.hold_pose(peace_sign())
        self.assertEqual(self.rotate_through(0, -30), "SWIPE_LEFT")
        self.rotate_through(-30, 0)
        self.hold_still(0)
        self.assertEqual(self.rotate_through(0, 30), "SWIPE_RIGHT")


class TestVerticalSwipes(GestureTestCase):
    """Raising and lowering the two fingers, for volume."""

    def move(self, start, end, seconds=0.30, frames=12, hold=True):
        """Move the hand vertically, in fractions of the frame."""

        if hold:
            self.hold_pose(pinching(cy=start))

        result = None
        for i in range(frames):
            cy = start + (end - start) * (i / (frames - 1))
            result = result or motion.detect_swipe(pinching(cy=cy), HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def test_raising_the_hand(self):
        """Screen y grows downwards, so a lift is a fall in y."""

        self.assertEqual(self.move(0.55, 0.30), "PINCH_UP")

    def test_lowering_the_hand(self):
        self.assertEqual(self.move(0.30, 0.55), "PINCH_DOWN")

    def test_a_small_lift_is_not_a_swipe(self):
        self.assertIsNone(self.move(0.50, 0.47))

    def test_a_slow_lift_is_not_a_swipe(self):
        """Lowering your hand to the desk must not change the volume."""

        self.assertIsNone(self.move(0.55, 0.30, seconds=4.0, frames=40))

    def test_the_hand_coming_back_down_is_not_a_second_swipe(self):
        self.assertEqual(self.move(0.55, 0.30), "PINCH_UP")
        self.assertIsNone(self.move(0.30, 0.55, hold=False))


class TestVolumeBothWays(GestureTestCase):
    """Raising and lowering have to work one after the other.

    Reported: up registered, down never did.  Height was judged by
    movement, and the way back from a raised hand is a lowered hand -- so
    every lowering was either the return from a raise, or would have had
    to start below where a hand naturally rests.  Height is now measured
    against where the hand was when the pose began.
    """

    def peace(self, cy):
        return pinching(cy=cy)

    def move(self, start, end, seconds=0.30, frames=12):
        result = None
        for i in range(frames):
            cy = start + (end - start) * (i / (frames - 1))
            result = result or motion.detect_swipe(self.peace(cy), HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def test_up_then_down(self):
        """Raise, come back, lower: two commands, opposite ways."""

        self.hold_pose(self.peace(0.50))
        self.assertEqual(self.move(0.50, 0.30), "PINCH_UP")
        self.move(0.30, 0.50)                      # back to the middle
        self.assertEqual(self.move(0.50, 0.70), "PINCH_DOWN")

    def test_down_then_up(self):
        self.hold_pose(self.peace(0.50))
        self.assertEqual(self.move(0.50, 0.70), "PINCH_DOWN")
        self.move(0.70, 0.50)
        self.assertEqual(self.move(0.50, 0.30), "PINCH_UP")

    def test_coming_back_does_not_fire(self):
        self.hold_pose(self.peace(0.50))
        self.assertEqual(self.move(0.50, 0.30), "PINCH_UP")
        self.assertIsNone(self.move(0.30, 0.50),
                          "the hand returning counted as a swipe down")

    def test_holding_it_up_fires_once(self):
        self.hold_pose(self.peace(0.50))
        self.assertEqual(self.move(0.50, 0.30), "PINCH_UP")

        for _ in range(30):
            self.assertIsNone(motion.detect_swipe(self.peace(0.30), HAND))
            self.clock.tick(0.03)


class TestRestingHeight(GestureTestCase):
    """Where the hand is held still is the height it is judged against.

    Reported twice: raising worked, lowering never did.  A hand comes into
    view from below, so a fixed starting point sits near the bottom of its
    range -- up is easy from there and down would mean leaving the frame.
    """

    def peace(self, cy):
        return pinching(cy=cy)

    def move(self, start, end, seconds=0.30, frames=12):
        result = None
        for i in range(frames):
            cy = start + (end - start) * (i / (frames - 1))
            result = result or motion.detect_swipe(self.peace(cy), HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def rest(self, cy, seconds=1.30, frames=26):
        for _ in range(frames):
            motion.detect_swipe(self.peace(cy), HAND)
            self.clock.tick(seconds / frames)

    def test_lowering_from_a_raised_hand(self):
        """Raise, hold a moment, lower: the second one is a swipe down."""

        self.hold_pose(self.peace(0.55))
        self.assertEqual(self.move(0.55, 0.35), "PINCH_UP")
        self.rest(0.35)
        self.assertEqual(self.move(0.35, 0.55), "PINCH_DOWN")

    def test_raising_twice_over(self):
        """Volume up, up again: each raise counts from the new height."""

        self.hold_pose(self.peace(0.70))
        self.assertEqual(self.move(0.70, 0.55), "PINCH_UP")
        self.rest(0.55)
        self.assertEqual(self.move(0.55, 0.40), "PINCH_UP")

    def test_a_hand_brought_in_low_can_still_go_down(self):
        """The case from the report: the pose is first seen low."""

        self.hold_pose(self.peace(0.60))
        self.rest(0.60)
        self.assertEqual(self.move(0.60, 0.78), "PINCH_DOWN")


class TestPuttingTheHandBack(GestureTestCase):
    """Putting the hand back where it was is not a gesture.

    Reported: raise for volume up, bring the hand back down, and the
    coming back counted as volume down -- and the same in reverse.  A
    brief pause at the top was enough to make the raised position the new
    resting height, which made the return a genuine downward move.
    """

    def peace(self, cy):
        return pinching(cy=cy)

    def move(self, start, end, seconds=0.30, frames=12):
        result = None
        for i in range(frames):
            cy = start + (end - start) * (i / (frames - 1))
            result = result or motion.detect_swipe(self.peace(cy), HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def pause(self, cy, seconds=0.35, frames=8):
        result = None
        for _ in range(frames):
            result = result or motion.detect_swipe(self.peace(cy), HAND)
            self.clock.tick(seconds / frames)
        return result

    def test_raise_pause_and_put_it_back(self):
        self.hold_pose(self.peace(0.60))
        self.assertEqual(self.move(0.60, 0.40), "PINCH_UP")
        self.pause(0.40)
        self.assertIsNone(self.move(0.40, 0.60),
                          "putting the hand back counted as a swipe down")

    def test_lower_pause_and_put_it_back(self):
        self.hold_pose(self.peace(0.45))
        self.assertEqual(self.move(0.45, 0.65), "PINCH_DOWN")
        self.pause(0.65)
        self.assertIsNone(self.move(0.65, 0.45),
                          "putting the hand back counted as a swipe up")

    def test_raising_twice_from_the_same_place(self):
        """Back where you started, so raising again is another swipe up."""

        self.hold_pose(self.peace(0.60))
        self.assertEqual(self.move(0.60, 0.40), "PINCH_UP")
        self.pause(0.40)
        self.move(0.40, 0.60)
        self.pause(0.60)
        self.assertEqual(self.move(0.60, 0.40), "PINCH_UP")

    def test_settling_at_a_new_height_does_move_the_reference(self):
        """Held there long enough, it becomes where the hand lives."""

        self.hold_pose(self.peace(0.60))
        self.assertEqual(self.move(0.60, 0.40), "PINCH_UP")
        self.pause(0.40, seconds=1.40, frames=28)
        self.assertEqual(self.move(0.40, 0.60), "PINCH_DOWN")


class TestPinch(GestureTestCase):
    """Thumb and finger together: the pose volume is made from.

    On its own pose rather than sharing the two-finger one, so a hand
    raised while seeking cannot be read as volume and the two no longer
    have to be told apart by direction alone.
    """

    def test_a_pinch_is_recognised(self):
        self.assertEqual(self.settle(pinching()), "PINCH")

    def test_a_fist_is_not_a_pinch(self):
        """A closed hand also lays the thumb against the index finger.

        Proximity alone called every fist a pinch; a fist tucks the
        fingertip into the palm, a pinch holds it out to meet the thumb.
        """

        self.assertEqual(self.settle(fist()), "FIST")

    def test_an_open_hand_is_not_a_pinch(self):
        self.assertEqual(self.settle(make_hand()), "OPEN_PALM")

    def test_two_fingers_are_not_a_pinch(self):
        self.assertEqual(self.settle(peace_sign()), "TWO_FINGER")

    def test_a_hand_aimed_at_the_camera_is_not_a_pinch(self):
        """Everything collapses together on screen at that angle.

        Measuring the gap in space rather than on screen is what fixed it.
        """

        for yaw in (-20, 0, 20):
            with self.subTest(yaw=yaw):
                vision.reset_state()
                aimed = make_hand(pitch=AT_CAMERA, yaw=math.radians(yaw))
                self.assertFalse(hand_state.is_pinching(aimed))

    def test_the_pinch_arms_the_vertical_gesture(self):
        self.assertEqual(gestures.pose_kind(pinching()), gestures.POSE_PINCH)

    def test_two_fingers_arm_the_sideways_one(self):
        self.assertEqual(gestures.pose_kind(peace_sign()), gestures.POSE_PEACE)

    def test_a_pinch_does_not_seek(self):
        """Turning a pinched hand sideways must not rewind."""

        self.hold_pose(pinching())

        result = None
        for i in range(12):
            turned = pinching(cx=0.50)
            result = result or motion.detect_swipe(turned, HAND)
            self.clock.tick(0.30 / 11)

        self.assertIsNone(result)


class TestPinchMistakenForOpenHand(GestureTestCase):
    """Reported from use: a pinch sometimes read as an open hand.

    Pinching with the other three fingers out leaves four fingers reading
    as extended, so a pinch that falls a little short of the pinch test
    lands on the open hand -- which is the gesture that turns SARV off.
    A pinch that goes unrecognised is a nuisance; a pinch that pauses the
    app is worse.
    """

    def loose_pinch(self, gap):
        """A pinch with the fingers not quite touching."""

        hand = pinching()
        tip = hand[hand_state.INDEX_TIP]
        hand[hand_state.THUMB_TIP] = Point(tip.x - gap, tip.y + gap * 0.6, 0.0)
        return hand

    def test_a_pinch_that_does_not_quite_touch_still_counts(self):
        self.assertEqual(self.settle(self.loose_pinch(0.030)), "PINCH")

    def test_a_pinch_too_loose_to_read_is_nothing_rather_than_open(self):
        """The important half: not recognised beats wrongly recognised."""

        self.assertEqual(self.settle(self.loose_pinch(0.060)), "UNKNOWN")

    def test_a_genuinely_open_hand_still_reads_as_one(self):
        self.assertEqual(self.settle(make_hand()), "OPEN_PALM")

    def test_the_gap_is_measured_the_same_way_at_any_angle(self):
        """Whatever the answer, it must not depend on the viewpoint."""

        face_on = hand_state.pinch_gap(pinching())

        for yaw in (-45, 0, 45):
            with self.subTest(yaw=yaw):
                turned = pinching(yaw=math.radians(yaw))
                self.assertAlmostEqual(face_on, hand_state.pinch_gap(turned),
                                       places=6)


class TestPeaceSignMistakenForPinch(GestureTestCase):
    """Reported: two fingers sometimes read as a pinch, and seeking stopped.

    Both halves of one fault.  A tucked thumb is half hidden behind the
    palm, so MediaPipe has to guess where its tip is, and the guess
    sometimes lands near the index finger.  Two fingers then look pinched
    -- and since the pinch arms volume rather than seeking, the wrist turn
    had nothing to arm and did nothing at all.
    """

    def misplaced_thumb(self, hand):
        """As the camera reports it when the thumb is behind the palm."""

        tip = hand[hand_state.INDEX_TIP]
        hand[hand_state.THUMB_TIP] = Point(tip.x + 0.01, tip.y + 0.01, 0.0)
        return hand

    def test_two_fingers_up_are_never_a_pinch(self):
        self.assertFalse(hand_state.is_pinching(
            self.misplaced_thumb(peace_sign())))

    def test_it_still_reads_as_two_fingers(self):
        self.assertEqual(
            self.settle(self.misplaced_thumb(peace_sign())), "TWO_FINGER")

    def test_it_still_arms_seeking(self):
        self.assertEqual(
            gestures.pose_kind(self.misplaced_thumb(peace_sign())),
            gestures.POSE_PEACE)

    def test_seeking_works_with_the_thumb_misread(self):
        """The symptom itself: the turn must still register."""

        def hand(roll):
            return self.misplaced_thumb(peace_sign(roll=math.radians(roll)))

        self.hold_pose(hand(-25))

        result = None
        for i in range(12):
            result = result or motion.detect_swipe(
                hand(-25 + 50 * (i / 11)), HAND)
            self.clock.tick(0.30 / 11)

        self.assertEqual(result, "SWIPE_RIGHT")

    def test_a_real_pinch_is_untouched(self):
        self.assertTrue(hand_state.is_pinching(pinching()))


class TestPinchHoweverItIsHeld(GestureTestCase):
    """A pinch counts whatever the other fingers are doing.

    Written after breaking exactly this.  Ruling out a pinch when the
    middle finger was out did fix two fingers being misread, and threw
    away every pinch made with the other fingers held out -- which is how
    a good many people pinch.  Nothing in the suite covered that, so
    nothing objected.
    """

    def curled(self):
        return pinching()

    def fingers_out(self):
        return make_hand(LOOSE, EXTENDED, EXTENDED, EXTENDED, pinch=True)

    def test_pinching_with_the_others_curled(self):
        self.assertEqual(self.settle(self.curled()), "PINCH")

    def test_pinching_with_the_others_held_out(self):
        self.assertEqual(self.settle(self.fingers_out()), "PINCH")

    def test_both_arm_the_volume_gesture(self):
        for hand in (self.curled(), self.fingers_out()):
            self.assertEqual(gestures.pose_kind(hand), gestures.POSE_PINCH)

    def test_an_open_hand_is_still_an_open_hand(self):
        """The other half of the same mistake: an open hand went missing
        behind a margin meant to protect it from pinches."""

        self.assertEqual(self.settle(make_hand()), "OPEN_PALM")

    def test_and_two_fingers_are_still_not_a_pinch(self):
        self.assertFalse(hand_state.is_pinching(peace_sign()))


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
