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
              cx=0.50, cy=0.50):
    """Build 21 landmarks for a hand with the given finger states."""

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
            # measurably nearer the camera than the last.
            points.append(Point(mx, my - 0.010, -0.035))
            points.append(Point(mx, my - 0.015, -0.070))
            points.append(Point(mx, my - 0.020, -0.105))

        else:  # CURLED
            points.append(Point(mx, my - 0.045, 0.000))
            points.append(Point(mx, my - 0.055, 0.020))
            points.append(Point(mx, my - 0.020, 0.030))

    assert len(points) == 21, len(points)
    return points


def two_finger_pointing(cx=0.50, cy=0.50):
    """The swipe pose: index and middle aimed at the camera.

    Aiming the fingers at the lens turns the whole hand, not just the
    fingers: the forearm now runs away from the camera, so the wrist sits
    further back and projects almost on top of the knuckles.  The knuckle
    row itself stays broadside to the camera, which is why it is the one
    measurement worth scaling by.
    """

    hand = make_hand(index=POINTING, middle=POINTING,
                     ring=CURLED, pinky=CURLED, cx=cx, cy=cy)
    hand[gd.WRIST] = Point(cx, cy + 0.01, 0.10)
    return hand


def edge_on_hand(cx=0.50):
    """A hand turned side-on, where the palm normal says nothing."""

    hand = make_hand(cx=cx)
    # Put wrist and both knuckles on one screen line.
    hand[gd.WRIST] = Point(cx, 0.60)
    hand[gd.INDEX_MCP] = Point(cx, 0.50)
    hand[gd.PINKY_MCP] = Point(cx, 0.40)
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

    def test_edge_on_palm_is_unknown_not_guessed(self):
        """The swipe pose must not be reported as a held gesture."""

        self.assertEqual(self.settle(edge_on_hand()), "UNKNOWN")

    def test_swipe_pose_does_not_fire_a_static_gesture(self):
        self.assertEqual(self.settle(two_finger_pointing()), "UNKNOWN")


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
    def sweep(self, start, end, seconds, frames=12, pose=True, hand=None):
        """Move the hand from start to end, returning any swipe detected."""

        result = None
        for i in range(frames):
            fraction = i / (frames - 1)
            cx = start + (end - start) * fraction
            landmarks = hand(cx) if hand else (
                two_finger_pointing(cx) if pose
                else make_hand(CURLED, CURLED, CURLED, CURLED, cx=cx))
            found = gd.detect_swipe(landmarks, HAND)
            result = result or found
            self.clock.tick(seconds / (frames - 1))
        return result

    def test_swipe_right(self):
        self.assertEqual(self.sweep(0.40, 0.58, 0.30), "SWIPE_RIGHT")

    def test_swipe_left(self):
        self.assertEqual(self.sweep(0.58, 0.40, 0.30), "SWIPE_LEFT")

    def test_slow_drift_is_not_a_swipe(self):
        """Resting a hand and moving it across the desk must do nothing."""

        self.assertIsNone(self.sweep(0.40, 0.58, 4.0, frames=40))

    def test_tiny_movement_is_not_a_swipe(self):
        self.assertIsNone(self.sweep(0.50, 0.52, 0.30))

    def test_wrong_pose_is_not_a_swipe(self):
        self.assertIsNone(self.sweep(0.40, 0.58, 0.30, pose=False))

    def test_wobble_is_not_a_swipe(self):
        """Shaking a hand covers distance without going anywhere."""

        result = None
        for i in range(14):
            cx = 0.50 + (0.05 if i % 2 else -0.05)
            result = result or gd.detect_swipe(two_finger_pointing(cx), HAND)
            self.clock.tick(0.025)
        self.assertIsNone(result)

    def test_a_bad_frame_does_not_lose_the_swipe(self):
        """The regression that made swipes feel broken.

        Position is recorded every frame, so one misclassified frame in the
        middle costs a sample, not the gesture.
        """

        result = None
        for i in range(12):
            cx = 0.40 + 0.18 * (i / 11)
            # Frame 6 comes back as an open hand.
            landmarks = (make_hand(cx=cx) if i == 6
                         else two_finger_pointing(cx))
            result = result or gd.detect_swipe(landmarks, HAND)
            self.clock.tick(0.30 / 11)
        self.assertEqual(result, "SWIPE_RIGHT")

    def test_cooldown_blocks_an_immediate_second_swipe(self):
        self.assertEqual(self.sweep(0.40, 0.58, 0.30), "SWIPE_RIGHT")
        self.assertIsNone(self.sweep(0.40, 0.58, 0.30))

    def test_swipe_allowed_again_after_cooldown(self):
        self.assertEqual(self.sweep(0.40, 0.58, 0.30), "SWIPE_RIGHT")
        self.clock.tick(gd.SWIPE_COOLDOWN + 0.1)
        self.assertEqual(self.sweep(0.40, 0.58, 0.30), "SWIPE_RIGHT")

    def test_distance_from_camera_does_not_matter(self):
        """Thresholds are in palm widths, so a smaller hand needs less travel."""

        def small(cx):
            return two_finger_pointing(cx=cx)

        far = self.sweep(0.40, 0.58, 0.30, hand=small)
        self.assertEqual(far, "SWIPE_RIGHT")

    def test_debug_state_is_populated(self):
        """The tuning overlay needs these every frame."""

        gd.detect_swipe(two_finger_pointing(), HAND)
        state = gd.debug_state()
        for key in ("pose", "armed", "travel", "speed", "agree"):
            self.assertIn(key, state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
