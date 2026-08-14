"""Gesture detection, tested with synthetic hands.

No camera is involved: each test builds the 21 landmarks MediaPipe would
produce for a given pose and feeds them in.  That cannot tell us whether a
real hand in real lighting is classified correctly -- only a person waving at
the webcam can -- but it does pin the logic: that a swipe survives a bad
frame, that a wobble is not a swipe, that an edge-on palm is reported as
unknown instead of guessed.

The frame is mirrored before detection, so MediaPipe labels a physical right
hand "Left".  These hands are built accordingly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import gesture_detection as gd


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


# --- hand building ---------------------------------------------------------

# Finger states
EXTENDED = "extended"      # straight, in the image plane
CURLED = "curled"          # folded toward the palm
POINTING = "pointing"      # straight, aimed at the camera


def make_hand(index=EXTENDED, middle=EXTENDED, ring=EXTENDED, pinky=EXTENDED,
              cx=0.50, cy=0.50, yaw=0.0):
    """Build 21 landmarks for a hand with the given finger states.

    ``yaw`` turns pointing fingers about the wrist, -1 (hard left) to +1
    (hard right).  The fingertips swing sideways and lose some of their
    depth as they go, which is what a real hand does when it turns away
    from the camera.
    """

    points = [Point(cx, cy + 0.18)] * 1          # wrist
    points = [Point(cx, cy + 0.18)]
    points += [Point(cx - 0.10, cy + 0.12)] * 4  # thumb, unused here

    layout = {
        "index": (cx - 0.06, cy, index),
        "middle": (cx - 0.02, cy - 0.01, middle),
        "ring": (cx + 0.02, cy, ring),
        "pinky": (cx + 0.06, cy + 0.01, pinky),
    }

    for name in ("index", "middle", "ring", "pinky"):
        mx, my, state = layout[name]
        points.append(Point(mx, my, 0.0))                       # mcp

        if state == EXTENDED:
            points.append(Point(mx, my - 0.05, 0.0))            # pip
            points.append(Point(mx, my - 0.09, 0.0))            # dip
            points.append(Point(mx, my - 0.13, 0.0))            # tip

        elif state == POINTING:
            # Foreshortened: barely moves on screen, but each joint is
            # measurably nearer the camera than the last.  Turning the
            # wrist swings the far joints sideways the most.
            depth = 1.0 - 0.4 * abs(yaw)
            points.append(Point(mx + yaw * 0.040, my - 0.010, -0.035 * depth))
            points.append(Point(mx + yaw * 0.080, my - 0.015, -0.070 * depth))
            points.append(Point(mx + yaw * 0.120, my - 0.020, -0.105 * depth))

        else:  # CURLED
            points.append(Point(mx, my - 0.045, 0.000))
            points.append(Point(mx, my - 0.055, 0.020))
            points.append(Point(mx, my - 0.020, 0.030))

    assert len(points) == 21, len(points)
    return points


def two_finger_pointing(cx=0.50, cy=0.50, yaw=0.0):
    """The swipe pose: index and middle aimed at the camera.

    Aiming the fingers at the lens turns the whole hand, not just the
    fingers: the forearm now runs away from the camera, so the wrist sits
    further back and projects almost on top of the knuckles.  The knuckle
    row itself stays broadside to the camera, which is why it is the one
    measurement worth scaling by.
    """

    hand = make_hand(index=POINTING, middle=POINTING,
                     ring=CURLED, pinky=CURLED, cx=cx, cy=cy, yaw=yaw)
    hand[gd.WRIST] = Point(cx, cy + 0.01, 0.10)
    return hand


def back_of_hand_fist(cx=0.50):
    """A punch: fingers curled, knuckles toward the camera.

    Mirroring the knuckle row flips the palm normal, which is what shows
    the back of the hand rather than the palm.
    """

    hand = make_hand(CURLED, CURLED, CURLED, CURLED, cx=cx)
    hand[gd.INDEX_MCP] = Point(cx + 0.06, 0.50)
    hand[gd.PINKY_MCP] = Point(cx - 0.06, 0.51)
    return hand


def pointing_at_camera(cx=0.50):
    """One finger aimed at the lens, the rest curled."""

    hand = make_hand(index=POINTING, middle=CURLED,
                     ring=CURLED, pinky=CURLED, cx=cx)
    hand[gd.WRIST] = Point(cx, 0.51, 0.10)
    return hand


HAND = "Left"  # what MediaPipe calls a right hand in a mirrored frame


class GestureTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        gd.time.time = self.clock
        gd.reset_state()
        gd.swipe_cooldown_until = 0.0

    def tearDown(self):
        import time as real_time
        gd.time.time = real_time.time


# --- static gestures -------------------------------------------------------

class TestStaticGestures(GestureTestCase):
    def settle(self, hand, frames=5):
        """Hold a pose long enough for the stabiliser to accept it."""

        result = "UNKNOWN"
        for _ in range(frames):
            result = gd.detect_gesture(hand, HAND)
        return result

    def test_open_palm(self):
        self.assertEqual(self.settle(make_hand()), "OPEN_PALM")

    def test_fist(self):
        self.assertEqual(
            self.settle(make_hand(CURLED, CURLED, CURLED, CURLED)), "FIST")

    def test_point(self):
        self.assertEqual(
            self.settle(make_hand(EXTENDED, CURLED, CURLED, CURLED)), "POINT")

    def test_two_finger(self):
        self.assertEqual(
            self.settle(make_hand(EXTENDED, EXTENDED, CURLED, CURLED)),
            "TWO_FINGER")

    def test_needs_several_frames(self):
        """One good frame is not a gesture; that is how flicker gets in."""

        self.assertEqual(gd.detect_gesture(make_hand(), HAND), "UNKNOWN")

    def test_punch_is_a_fist(self):
        """A punch shows the camera its knuckles, not its palm.

        Requiring a visible palm rejected this before the fingers were
        even counted, which is why a punch registered as nothing.
        """

        self.assertEqual(self.settle(back_of_hand_fist()), "FIST")

    def test_pointing_at_the_camera_is_a_point(self):
        """The other gesture the palm check used to swallow.

        An index finger aimed at the lens is foreshortened into a short
        cluster of landmarks, so it reads as curled by joint angle; it is
        recognised by depth instead -- tip nearer than knuckle.
        """

        self.assertEqual(self.settle(pointing_at_camera()), "POINT")

    def test_gun_pose_reads_as_two_finger(self):
        """Accepted consequence of dropping the orientation check.

        Two fingers really are out, so it is not wrong -- and swipes come
        from detect_swipe, which is a separate call.
        """

        self.assertEqual(self.settle(two_finger_pointing()), "TWO_FINGER")


# --- pose recognition ------------------------------------------------------

class TestTwoFingerPose(GestureTestCase):
    def test_recognised_pointing_at_camera(self):
        self.assertTrue(gd.is_two_finger_pose(two_finger_pointing()))

    def test_recognised_palm_on_too(self):
        """The hand rotates as the arm moves; both orientations must count."""

        self.assertTrue(gd.is_two_finger_pose(
            make_hand(EXTENDED, EXTENDED, CURLED, CURLED)))

    def test_open_hand_is_not_the_pose(self):
        self.assertFalse(gd.is_two_finger_pose(make_hand()))

    def test_fist_is_not_the_pose(self):
        self.assertFalse(gd.is_two_finger_pose(
            make_hand(CURLED, CURLED, CURLED, CURLED)))


# --- swipes ----------------------------------------------------------------

class TestSwipe(GestureTestCase):
    """The gesture: wrist held still, two fingers turning left or right."""

    def turn(self, start, end, seconds, frames=12, pose=True, cx=0.50):
        """Rotate the wrist from one aim to another, at a fixed position."""

        result = None
        for i in range(frames):
            fraction = i / (frames - 1)
            yaw = start + (end - start) * fraction
            landmarks = (
                two_finger_pointing(cx=cx, yaw=yaw) if pose
                else make_hand(CURLED, CURLED, CURLED, CURLED, cx=cx))
            found = gd.detect_swipe(landmarks, HAND)
            result = result or found
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
        """Lowering your hand slowly must not seek the video."""

        self.assertIsNone(self.turn(-0.6, 0.6, 4.0, frames=40))

    def test_small_turn_is_not_a_swipe(self):
        self.assertIsNone(self.turn(0.0, 0.1, 0.30))

    def test_wrong_pose_is_not_a_swipe(self):
        self.assertIsNone(self.turn(-0.6, 0.6, 0.30, pose=False))

    def test_wobble_is_not_a_swipe(self):
        """A shaking hand turns a lot without turning anywhere."""

        result = None
        for i in range(14):
            yaw = 0.5 if i % 2 else -0.5
            result = result or gd.detect_swipe(
                two_finger_pointing(yaw=yaw), HAND)
            self.clock.tick(0.025)
        self.assertIsNone(result)

    def test_carrying_the_gun_pose_across_is_not_a_swipe(self):
        """Aiming at the camera and reaching for the keyboard must do nothing.

        Aim is measured relative to the wrist, so moving the hand without
        turning it reads as no movement at all.  The peace sign is treated
        differently -- see TestPeaceSignSwipe -- because a slide is how you
        naturally swipe a hand that is held up rather than aimed.
        """

        result = None
        for i in range(12):
            cx = 0.30 + 0.40 * (i / 11)
            result = result or gd.detect_swipe(
                two_finger_pointing(cx=cx, yaw=0.0), HAND)
            self.clock.tick(0.30 / 11)
        self.assertIsNone(result)

    def test_a_bad_frame_does_not_lose_the_swipe(self):
        """The regression that made swipes feel broken.

        Aim is recorded every frame, so one misclassified frame in the
        middle costs a sample, not the gesture.  This matters more for a
        turn than a slide: a hand rotating away from the camera passes
        through angles that are genuinely hard to classify.
        """

        result = None
        for i in range(12):
            yaw = -0.6 + 1.2 * (i / 11)
            # Frame 6 comes back as an open hand.
            landmarks = (make_hand(cx=0.50) if i == 6
                         else two_finger_pointing(yaw=yaw))
            result = result or gd.detect_swipe(landmarks, HAND)
            self.clock.tick(0.30 / 11)
        self.assertEqual(result, "SWIPE_RIGHT")

    def test_cooldown_blocks_an_immediate_second_swipe(self):
        self.assertEqual(self.turn(-0.6, 0.6, 0.30), "SWIPE_RIGHT")
        self.assertIsNone(self.turn(-0.6, 0.6, 0.30))

    def test_swipe_allowed_again_after_cooldown(self):
        self.assertEqual(self.turn(-0.6, 0.6, 0.30), "SWIPE_RIGHT")
        self.clock.tick(gd.SWIPE_COOLDOWN + 0.1)
        self.assertEqual(self.turn(-0.6, 0.6, 0.30), "SWIPE_RIGHT")

    def test_position_in_frame_does_not_matter(self):
        """The same turn works wherever you hold your hand."""

        for cx in (0.25, 0.50, 0.75):
            with self.subTest(cx=cx):
                gd.reset_state()
                gd.swipe_cooldown_until = 0.0
                self.assertEqual(
                    self.turn(-0.6, 0.6, 0.30, cx=cx), "SWIPE_RIGHT")

    def test_debug_state_is_populated(self):
        """The tuning overlay needs these every frame."""

        gd.detect_swipe(two_finger_pointing(), HAND)
        state = gd.debug_state()
        for key in ("pose", "armed", "aim", "turn", "slide", "speed", "agree"):
            self.assertIn(key, state)


def peace_sign(cx=0.50, tilt=0.0):
    """Two fingers held up, palm toward the camera.

    ``tilt`` leans the fingers sideways, which is the other way to swipe
    this pose.
    """

    hand = make_hand(EXTENDED, EXTENDED, CURLED, CURLED, cx=cx)

    if tilt:
        for mcp in (5, 9):
            for joint in range(mcp + 1, mcp + 4):
                lean = (joint - mcp) * 0.045 * tilt
                hand[joint] = Point(hand[joint].x + lean,
                                    hand[joint].y,
                                    hand[joint].z)
    return hand


class TestPeaceSignSwipe(GestureTestCase):
    """Two fingers up, palm to the camera: slid or tilted across."""

    def slide(self, start, end, seconds, frames=12):
        result = None
        for i in range(frames):
            cx = start + (end - start) * (i / (frames - 1))
            result = result or gd.detect_swipe(peace_sign(cx=cx), HAND)
            self.clock.tick(seconds / (frames - 1))
        return result

    def test_pose_is_recognised(self):
        self.assertEqual(gd.two_finger_pose_kind(peace_sign()), gd.POSE_PEACE)

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
            tilt = -1.0 + 2.0 * (i / 11)
            result = result or gd.detect_swipe(peace_sign(tilt=tilt), HAND)
            self.clock.tick(0.30 / 11)
        self.assertEqual(result, "SWIPE_RIGHT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
