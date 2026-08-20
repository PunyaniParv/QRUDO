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
        # Held out to the side the way real hands hold it: live open
        # palms measure the thumb 0.5+ palm-lengths from the index
        # knuckle and 0.35+ from the index TIP, and the strict palm
        # tests lean on both distances.
        thumb_tip = Point(cx - 0.160, cy - 0.010)

    points += [Point(cx - 0.085, cy + 0.140),
               Point(cx - 0.110, cy + 0.100),
               Point(cx - 0.140, cy + 0.045),
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

    def test_open_palm_only_facing_the_camera(self):
        """Angled is fine; side-on is not a palm shown to anybody.

        It used to read from any side but behind, and an open hand
        side-on is what the camera sees of someone reaching past it --
        with the brightness pose armed by the reach.
        """

        facing = [math.radians(degrees) for degrees in (-45, -20, 0, 20, 45)]

        self.check_all_views(make_hand, "OPEN_PALM",
                             yaws=facing, pitches=self.PALM_PITCHES)

    def test_open_palm_is_not_read_side_on(self):
        edge_on = [math.radians(degrees) for degrees in (-90, 90)]

        self.check_all_views(make_hand, "UNKNOWN",
                             yaws=edge_on, pitches=self.PALM_PITCHES)

        self.assertIsNone(
            gestures.pose_kind(make_hand(yaw=math.radians(90))),
            "an edge-on palm must not arm the brightness lifts")

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

        self.hold_pose(gun(turn=-0.6))

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
    it landed on "open", which is the gesture that turns QRUDO off.
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


class TestSlackIsNeitherOutNorFolded(GestureTestCase):
    """A hand at rest curls its fingers in order -- index straightest,
    pinky deepest -- so a single line always cuts through the middle of
    somebody's resting hand, and some depth of slack leaves exactly the
    index above it.  That read as POINT, and one slacker finger as the
    swipe pose, armed by a hand doing nothing.

    With the folded line drawn, a slack finger is neither out nor down,
    and a pattern may claim it as neither.  The line ships equal to the
    out-line -- the old single-line behaviour -- and only a calibration
    that measured the separation lowers it, so these tests run as a
    calibrated hand.
    """

    def setUp(self):
        super().setUp()
        self.was = hand_state.FOLDED_RATIO
        hand_state.FOLDED_RATIO = 0.65

    def tearDown(self):
        hand_state.FOLDED_RATIO = self.was
        super().tearDown()

    def test_the_index_straightest_of_a_slack_hand_is_not_point(self):
        self.assertEqual(
            self.settle(make_hand(EXTENDED, LOOSE, LOOSE, LOOSE)),
            "UNKNOWN")

    def test_one_slack_finger_deeper_is_not_the_swipe_pose(self):
        hand = make_hand(EXTENDED, EXTENDED, LOOSE, LOOSE)

        self.assertEqual(self.settle(hand), "UNKNOWN")
        self.assertIsNone(gestures.pose_kind(hand))

    def test_fingers_folded_on_purpose_still_read(self):
        self.assertEqual(
            self.settle(make_hand(EXTENDED, CURLED, CURLED, CURLED)),
            "POINT")

        vision.reset_state()
        hand = make_hand(EXTENDED, EXTENDED, CURLED, CURLED)

        self.assertEqual(self.settle(hand), "TWO_FINGER")
        self.assertEqual(gestures.pose_kind(hand), gestures.POSE_TWO_FINGER)

    def test_the_pose_as_people_actually_make_it(self):
        """Pinky tucked, ring only half-held.  The calibrated fold line
        comes from a recording where the ring was folded right down for
        the prompt, and asking for that depth in use refused the pose as
        made casually -- which is how it is made.  The pinky carries the
        deliberateness; the ring only has to be short of out.
        """

        hand = make_hand(EXTENDED, EXTENDED, LOOSE, CURLED)

        self.assertEqual(self.settle(hand), "TWO_FINGER")
        self.assertEqual(gestures.pose_kind(hand), gestures.POSE_TWO_FINGER)


class TestTheCliff(unittest.TestCase):
    """The hand that forced the relative rule, in its own numbers.

    Logged live: the pose read index 1.00, middle 1.00, ring 0.70,
    pinky 0.70 -- and the pinky at rest read 0.73, one hundredth above
    the pinky held down.  No recorded bar fits in a hundredth.  What
    separates the two is shape: at rest each finger is a small step
    below the last (that pinky rested 0.16 under its up-fingers), and
    in the pose the down-fingers sit off a cliff (0.30 under them).
    """

    LIVE_POSE = {"index": 1.0, "middle": 1.0, "ring": 0.70, "pinky": 0.70}
    LIVE_REST = {"index": 0.92, "middle": 0.90, "ring": 0.85, "pinky": 0.74}

    DEEP_REST = {"index": 0.85, "middle": 0.84, "ring": 0.79, "pinky": 0.68}

    def out(self, spans):
        # As the user's calibration reads them: out above 0.801.
        return {name: span > 0.801 for name, span in spans.items()}

    def test_the_pose_that_was_refused_now_reads(self):
        self.assertTrue(gestures._two_up(self.out(self.LIVE_POSE),
                                         self.LIVE_POSE))

    def test_the_resting_hand_still_does_not(self):
        self.assertFalse(gestures._two_up(self.out(self.LIVE_REST),
                                          self.LIVE_REST))

    def test_a_deeper_rest_does_not_either(self):
        """The gradient slid down until the pinky ducked under the old
        absolute fold line -- the hole the cliff exists to close.  The
        steps between the fingers slide with the hand, so the cliff
        refuses rest at every depth at once."""

        self.assertFalse(gestures._two_up(self.out(self.DEEP_REST),
                                          self.DEEP_REST))

    def test_the_rest_step_is_under_the_cliff_with_room(self):
        step = (min(self.LIVE_REST["index"], self.LIVE_REST["middle"])
                - self.LIVE_REST["pinky"])

        self.assertLess(step + 0.05, hand_state.TWO_CLIFF,
                        "rest plus noise must not reach the cliff")


class TestTheRestSignatureInformsButDoesNotGate(GestureTestCase):
    """The signature vetoed briefly, and on a hand whose rest is
    naturally open the ball around it swallowed the casual versions of
    real poses: a swipe pose with its spare fingers held lazily sat
    entirely inside it, and a palm shown at resting height read as
    nothing until the hand was first carried somewhere the signature did
    not reach.  The folded lines guard the patterns structurally and the
    open-hand line is the measured split between rest and palm; the
    signature's remaining job is to say so in the tuning overlay.
    """

    def tearDown(self):
        hand_state.REST_SIGNATURE = None
        super().tearDown()

    def test_a_pose_matching_the_signature_still_reads(self):
        hand = make_hand()  # a properly open palm

        hand_state.REST_SIGNATURE = dict(hand_state.finger_span(hand))

        self.assertEqual(self.settle(hand), "OPEN_PALM")
        self.assertEqual(gestures.pose_kind(hand), gestures.POSE_OPEN_PALM)

    def test_the_tuning_overlay_mentions_the_match(self):
        hand = make_hand()

        hand_state.REST_SIGNATURE = dict(hand_state.finger_span(hand))
        self.settle(hand)

        self.assertIn("at rest", gestures.explain(hand, HAND))

    def test_without_a_signature_nothing_changes(self):
        self.assertEqual(self.settle(make_hand()), "OPEN_PALM")


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


class TestATurnCountsTheSameWhereverItStarts(GestureTestCase):
    """Reported as swipe right not being detected.

    Which way the fingers lean was measured as the sine of the angle,
    and a sine flattens out.  Forty degrees of wrist read 0.67 from
    upright and 0.26 from a wrist already turned 55 -- the same movement,
    a third of the reading.  So on a hand that habitually rests leaning
    one way, turning further that way was measured as barely happening,
    while turning back the other way had the whole range to itself.

    That is not a threshold that can be tuned around.  It is one
    direction being worth less than the other.
    """

    def peace(self, roll):
        return make_hand(EXTENDED, EXTENDED, CURLED, CURLED,
                         roll=math.radians(roll))

    def turn(self, start, degrees, seconds=0.30, frames=14):
        for _ in range(24):
            motion.detect_swipe(self.peace(start), HAND)
            self.clock.tick(0.04)

        fired = None

        for i in range(frames):
            share = i / (frames - 1)
            fired = fired or motion.detect_swipe(
                self.peace(start + degrees * share), HAND)
            self.clock.tick(seconds / (frames - 1))

        return fired

    def test_the_same_turn_reads_the_same_wherever_it_starts(self):
        change = []

        for start in (-55, -20, 0, 20, 55):
            here = hand_state.pointing_direction(self.peace(start))
            there = hand_state.pointing_direction(self.peace(start + 40))
            change.append(abs(there - here))

        self.assertAlmostEqual(min(change), max(change), places=2)

    def test_both_ways_from_a_hand_that_rests_leaning_right(self):
        for degrees, wanted in ((40, "SWIPE_RIGHT"), (-40, "SWIPE_LEFT")):
            with self.subTest(turning=degrees):
                self.setUp()

                self.assertEqual(self.turn(45, degrees), wanted)

    def test_both_ways_from_a_hand_that_rests_leaning_left(self):
        for degrees, wanted in ((40, "SWIPE_RIGHT"), (-40, "SWIPE_LEFT")):
            with self.subTest(turning=degrees):
                self.setUp()

                self.assertEqual(self.turn(-45, degrees), wanted)

    def test_a_hand_aimed_at_the_camera_is_still_guarded(self):
        """The reach floor is what stops a hand collapsing on screen from
        reading as an enormous turn, and it is still there."""

        aimed = make_hand(EXTENDED, EXTENDED, CURLED, CURLED,
                          pitch=math.radians(90))

        self.assertLessEqual(abs(hand_state.pointing_direction(aimed)),
                             math.pi / 2)


class TestAMovementSeenFromAcrossTheRoom(GestureTestCase):
    """A small movement, read with the error a distant hand comes with.

    Reported as two fingers raised not registering from far away.  The
    pose held; the movement was refused, and what refused it was the
    measure of how directly it travelled.  That was a sum of every
    frame's step, so every frame's error went into it -- and the further
    away the hand, the larger a share of each step the error is.  A clean
    raise came out looking like a hand shaking.
    """

    def peace(self, dy=0.0, scale=0.022):
        hand = make_hand(EXTENDED, EXTENDED, CURLED, CURLED)
        wrist = hand[0]
        factor = scale / hand_state.hand_scale(hand)

        return [Point(wrist.x + (point.x - wrist.x) * factor,
                      wrist.y + (point.y - wrist.y) * factor + dy,
                      wrist.z + (point.z - wrist.z) * factor)
                for point in hand]

    def noisy(self, hand, share, rng, scale=0.022):
        return [Point(point.x + rng.gauss(0, scale * share),
                      point.y + rng.gauss(0, scale * share),
                      point.z + rng.gauss(0, scale * share))
                for point in hand]

    def raise_it(self, share, scale=0.022, seed=5):
        rng = random.Random(seed)

        for _ in range(24):
            hand = self.noisy(self.peace(scale=scale), share, rng, scale)
            gestures.detect_gesture(hand, HAND)
            motion.detect_swipe(hand, HAND)
            self.clock.tick(1 / 30)

        fired = None

        for i in range(14):
            hand = self.noisy(self.peace(-scale * 1.6 * i / 13, scale),
                              share, rng, scale)
            gestures.detect_gesture(hand, HAND)
            fired = fired or motion.detect_swipe(hand, HAND)
            self.clock.tick(0.35 / 13)

        return fired

    def test_a_raise_survives_the_error_a_distant_hand_comes_with(self):
        for share in (0.02, 0.05):
            with self.subTest(error=share):
                self.setUp()

                self.assertEqual(self.raise_it(share), "SWIPE_UP")

    def test_and_most_of_the_way_past_it(self):
        self.assertEqual(self.raise_it(0.08), "SWIPE_UP")

    def test_directness_is_not_a_sum_of_every_frame(self):
        """The measure that was refusing them.

        A ramp with error on every frame is a movement; frame by frame it
        scored as a shake, because the error was added up and the signal
        was not.
        """

        ramp = [i / 13 for i in range(14)]
        noisy = [value + (0.06 if i % 2 else -0.06)
                 for i, value in enumerate(ramp)]
        times = [i / 30 for i in range(14)]

        _, _, direct = motion.measure(times, noisy)

        self.assertGreater(direct, motion.SWIPE_CONSISTENCY)

    def test_but_a_movement_that_arrives_nowhere_still_scores_nothing(self):
        values = [0, .5, 1.0, 1.4, 1.0, .5, 0.05]
        times = [i / 30 for i in range(len(values))]

        _, _, direct = motion.measure(times, values)

        self.assertLess(direct, motion.SWIPE_CONSISTENCY)


class TestTheFloorRisesWithTheNoise(GestureTestCase):
    """A movement must clear the reading's own jitter, not only the bar.

    The calibrated bar assumes the camera the calibration was made on.
    On a worse one -- dim light, a poor webcam, a hand at the edge of
    readability -- the jitter grows toward the bar, and a still hand
    starts to fire.  So the bar is joined by a floor of four times the
    window's median frame-to-frame step, which is the jitter itself.

    The floor can only raise the bar, never lower it: on a good camera it
    sits far beneath and decides nothing, which is what makes it safe to
    apply everywhere.
    """

    SCALE = 0.022          # a hand about two metres off

    def hand(self, dy, share, rng):
        base = make_hand(EXTENDED, EXTENDED, CURLED, CURLED)
        wrist = base[0]
        k = self.SCALE / hand_state.hand_scale(base)
        sigma = self.SCALE * share

        return [Point(wrist.x + (p.x - wrist.x) * k + rng.gauss(0, sigma),
                      wrist.y + (p.y - wrist.y) * k + dy + rng.gauss(0, sigma),
                      wrist.z + (p.z - wrist.z) * k + rng.gauss(0, sigma))
                for p in base]

    def watch(self, share, seconds=12, movement=None, seed=11):
        rng = random.Random(seed)
        fired = []

        for i in range(int(seconds * 30)):
            dy = movement(i) if movement else 0.0
            hand = self.hand(dy, share, rng)
            gestures.detect_gesture(hand, HAND)
            got = motion.detect_swipe(hand, HAND)

            if got:
                fired.append(got)

            self.clock.tick(1 / 30)

        return fired

    def lowered_bar(self, to=0.36):
        """A gentle calibration, which is where the bar alone failed."""

        motion.SWIPE_LIFT = to
        motion.SWIPE_TURN = 0.30 * (to / 0.60)

    def setUp(self):
        super().setUp()
        self.was = (motion.SWIPE_LIFT, motion.SWIPE_TURN)

    def tearDown(self):
        motion.SWIPE_LIFT, motion.SWIPE_TURN = self.was
        super().tearDown()

    def test_a_still_hand_on_an_awful_camera_fires_nothing(self):
        """Even against a bar a gentle calibration would set.

        This exact case fired two or three times in twenty seconds on the
        bar alone.
        """

        self.lowered_bar()

        self.assertEqual(self.watch(0.08), [])

    def test_and_nothing_when_the_noise_doubles_again(self):
        self.lowered_bar()

        self.assertEqual(self.watch(0.16), [])

    def test_a_real_raise_still_clears_its_own_floor(self):
        """The floor is a fraction of the movement's own size, so it can
        refuse a still hand without refusing a moving one."""

        def raising(i):
            if i < 24:
                return 0.0

            return -self.SCALE * 1.6 * min(1.0, (i - 24) / 13)

        self.assertIn("SWIPE_UP", self.watch(0.05, seconds=3, movement=raising))

    def test_on_a_good_camera_it_decides_nothing(self):
        """The bar is 8x to 33x the wander there; the floor sits far
        below it, and the full suite passing unchanged is the proof."""

        def raising(i):
            if i < 24:
                return 0.0

            return -self.SCALE * 1.6 * min(1.0, (i - 24) / 13)

        self.assertIn("SWIPE_UP", self.watch(0.02, seconds=3, movement=raising))


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

    def test_a_waggle_that_trends_one_way_does_count(self):
        """A trade, and worth naming.

        This used to be refused, by a directness measured frame by frame
        -- which also refused a clean movement seen from across the room,
        where each frame's error is a large share of each step.  Measured
        across stretches instead, a hand that wandered but ended up a
        long way to one side reads as having gone there, which is not an
        unreasonable thing to say about it.
        """

        self.assertEqual(
            self.play(rolls=[-25, 10, -20, 12, -18, 15, 22], seconds=0.40),
            "SWIPE_RIGHT")

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

    #: Two pixels of error, at the width the camera is now asked for.
    BLUR = 2.0 / 1760

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

    def test_the_floor_sits_where_the_poses_fail_and_not_before(self):
        """It is the range limit and nothing else is.

        Reported as stopping at about 1.3 metres, which is exactly the
        hand size the floor was set at -- so the vision was being cut off
        well before it was failing.

        An open hand is excepted, as it is everywhere: it asks the most
        of the reading, and on the shipped thresholds -- which demand a
        good deal more of it than a measured one does -- it is unreliable
        at the floor.  Nothing is misread as a result; brightness simply
        wants you nearer.
        """

        for wanted in ("FIST", "POINT", "TWO_FINGER"):
            with self.subTest(pose=wanted, at="the floor"):
                self.assertGreater(
                    self.accuracy(wanted, hand_state.MIN_HAND_ON_SCREEN), 0.85)

            with self.subTest(pose=wanted, at="well inside it"):
                self.assertEqual(self.accuracy(wanted, 0.030), 1.0)

    def test_lowering_it_costs_nothing_close_up(self):
        """Which is the whole reason it can be lowered at all: it decides
        which hands are too small to bother with, and says nothing about
        a hand near the camera."""

        for wanted in self.POSES:
            for scale in (0.25, 0.12, 0.06):
                with self.subTest(pose=wanted, scale=scale):
                    self.assertEqual(self.accuracy(wanted, scale), 1.0)

    def test_an_open_hand_is_the_shortest_ranged_of_them(self):
        """Worth stating, because brightness rides on it.

        It asks the most of the reading: every finger has to be not
        merely out but properly straight, and one finger misread by
        enough is all it takes.  That strictness is not removable -- a
        hand at rest is told from an open one by the pinky alone, on the
        calibration that prompted this -- so an open hand simply has to
        be nearer.
        """

        self.assertEqual(self.accuracy("OPEN_PALM", 0.04), 1.0)
        self.assertLess(self.accuracy("OPEN_PALM", 0.022), 0.50)

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


class TestTheWayInIsNotAGesture(GestureTestCase):
    """A hand arrives in frame from an edge, and the way in from an edge
    is a movement -- entering from below is a raise, made in whatever
    pose the hand happens to arrive in.  The neutral height used to be
    taken from the hand's first visible moments, mid-flight, so the
    entrance performed the gesture.  Nothing may fire until the hand has
    genuinely stopped once; the stop is what makes the next movement a
    departure rather than a continuation.
    """

    PALM = (EXTENDED, EXTENDED, EXTENDED, EXTENDED)
    TWO = (EXTENDED, EXTENDED, CURLED, CURLED)

    def enter(self, fingers, cy_from, cy_to, seconds=0.5, fps=30):
        fired = []
        frames = max(2, round(seconds * fps))

        for i in range(frames):
            cy = cy_from + (cy_to - cy_from) * (i / (frames - 1))
            got = motion.detect_swipe(make_hand(*fingers, cy=cy), HAND)

            if got:
                fired.append(got)

            self.clock.tick(1.0 / fps)

        return fired

    def hold(self, fingers, cy, seconds=0.3):
        self.enter(fingers, cy, cy, seconds)

    def test_entering_from_below_is_not_a_raise(self):
        self.assertEqual(self.enter(self.PALM, 0.95, 0.45), [])

    def test_entering_from_above_is_not_a_lower(self):
        self.assertEqual(self.enter(self.PALM, 0.05, 0.55), [])

    def test_the_swipe_pose_entering_is_not_volume(self):
        self.assertEqual(self.enter(self.TWO, 0.95, 0.45), [])

    def test_after_stopping_the_same_hand_gestures_freely(self):
        self.assertEqual(self.enter(self.PALM, 0.95, 0.55), [])
        self.hold(self.PALM, 0.55)

        self.assertEqual(self.enter(self.PALM, 0.55, 0.35, seconds=0.3),
                         ["PALM_UP"])

    def test_a_sideways_entrance_leaves_lifts_ready(self):
        """Entering across the frame never moved vertically, so the
        vertical axis was at rest the whole way in."""

        fired = []

        for i in range(15):
            cx = 0.05 + (0.50 - 0.05) * (i / 14)
            got = motion.detect_swipe(make_hand(*self.PALM, cx=cx), HAND)

            if got:
                fired.append(got)

            self.clock.tick(1.0 / 30)

        self.assertEqual(fired, [], "the way in fired something")
        self.assertEqual(self.enter(self.PALM, 0.55, 0.35, seconds=0.3),
                         ["PALM_UP"])


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

    def test_down_needs_no_more_travel_than_up_whatever_the_bars(self):
        """Reported as the palm having to go much lower for down than up.

        A lower carries wrist roll that a raise does not, and the
        crosstalk shares were normalised by the calibrated bars -- so a
        calibration that tightened the turn bar re-scored every ordinary
        drop as mostly turn, and only a much deeper drop outweighed it.
        The shares are in fixed units now; a calibration must not be
        able to make one direction dearer than the other.
        """

        bars = (motion.SWIPE_TURN, motion.SWIPE_LIFT)
        motion.SWIPE_TURN, motion.SWIPE_LIFT = 0.248, 0.60  # a real one

        try:
            self.assertEqual(self.move(self.PALM, rise=0.20), "PALM_UP")
            self.setUp()
            self.assertEqual(self.move(self.PALM, rise=-0.20, roll=-15),
                             "PALM_DOWN")
        finally:
            motion.SWIPE_TURN, motion.SWIPE_LIFT = bars

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

    def frame(self, hand):
        """One frame, taken as the app takes it: read once, then asked.

        A test that drove pose_kind alone used to feed the memory by
        accident, which is the double-counting that made five readings
        cover under two frames.
        """

        gestures.observe(hand)

        return gestures.pose_kind(hand)

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
            self.frame(self.two())

        for _ in range(2):
            self.assertIsNotNone(self.frame(self.two(ring=EXTENDED)))

        self.assertIsNotNone(self.frame(self.two()))

    def test_a_finger_that_stays_out_is_believed(self):
        """And this is the same reading that does not stop: three fingers.

        Kept apart from two, so it stays available to mean something of
        its own.
        """

        for _ in range(6):
            self.frame(self.two())

        for _ in range(4):
            self.frame(self.two(ring=EXTENDED))

        self.assertIsNone(self.frame(self.two(ring=EXTENDED)))

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


class TestTheRestBeforeThePoseCounts(GestureTestCase):
    """Reported as palm up doing nothing and palm down firing instead.

    The rest a raise departs from is mostly made before the pose is:
    the hand sits loose at the bottom, makes the palm, and goes up in
    one motion.  Stops were registered only while the pose was armed,
    so that rest was never seen -- the raise was refused for departing
    from no stop, the pause at its top became the first stop on record,
    and bringing the hand back down was the first eligible movement.
    Palm up did nothing; putting the hand away turned the brightness
    down.
    """

    LOOSE_HAND = (LOOSE, LOOSE, LOOSE, LOOSE)
    PALM = (EXTENDED, EXTENDED, EXTENDED, EXTENDED)
    TWO = (EXTENDED, EXTENDED, CURLED, CURLED)

    def rest_loose(self, cy, seconds=0.5, frames=15):
        for _ in range(frames):
            motion.detect_swipe(make_hand(*self.LOOSE_HAND, cy=cy), HAND)
            self.clock.tick(seconds / frames)

    def move(self, fingers, cy_from, cy_to, seconds=0.30, frames=12):
        fired = []

        for i in range(frames):
            cy = cy_from + (cy_to - cy_from) * (i / (frames - 1))
            got = motion.detect_swipe(make_hand(*fingers, cy=cy), HAND)

            if got:
                fired.append(got)

            self.clock.tick(seconds / (frames - 1))

        return fired

    def test_a_raise_straight_from_a_loose_rest_fires(self):
        """Make the pose and go up in one motion, as people actually do."""

        self.rest_loose(0.55)

        self.assertEqual(self.move(self.PALM, 0.55, 0.35), ["PALM_UP"])

    def test_the_two_finger_raise_works_the_same_way(self):
        self.rest_loose(0.55)

        self.assertEqual(self.move(self.TWO, 0.55, 0.35), ["SWIPE_UP"])

    def test_putting_the_hand_back_fires_nothing(self):
        """The raise fires; the way back must not fire its opposite."""

        self.rest_loose(0.55)
        self.assertEqual(self.move(self.PALM, 0.55, 0.35), ["PALM_UP"])
        self.move(self.PALM, 0.35, 0.35, seconds=0.2, frames=6)

        self.assertEqual(self.move(self.PALM, 0.35, 0.55), [])


class TestAJitteryAimStillRests(GestureTestCase):
    """At a distance the aim wobbles at rest, and a rest must still count.

    The aim is an asin of a noisy ratio, so far from the camera its
    wobble alone can exceed the fixed stillness bar.  An aim that never
    rests refuses every turn -- a turn must depart from a rested aim --
    which is why left and right died at range while up and down, whose
    stillness already allowed for the reading's own jitter, survived
    it.  The same allowance now applies to both.
    """

    def test_a_turn_fires_though_the_resting_aim_wobbles(self):
        wobble = (0.0, 4.0, -4.0)

        for i in range(18):
            motion.detect_swipe(
                peace_sign(roll=math.radians(wobble[i % 3])), HAND)
            self.clock.tick(1.0 / 30)

        result = None

        for i in range(12):
            result = result or motion.detect_swipe(
                peace_sign(roll=math.radians(-35 * (i / 11))), HAND)
            self.clock.tick(0.30 / 11)

        self.assertEqual(result, "SWIPE_LEFT")


class TestOnlyTheRealShapeCounts(GestureTestCase):
    """Only a fist is a fist, and only a whole palm is a palm.

    Reported as the classifier being too generous: a thumbs-up fired
    the fist's action, and four fingers with the thumb tucked across
    the palm fired the palm's.  Both shapes mean something else to the
    person making them, so they must mean nothing to the built-ins.
    The thumb tests only apply to a hand near enough for its thumb to
    be read honestly; the range grid in test_capability_floor pins
    that far-off hands still carry the poses on four fingers alone.
    """

    def thumbs_up(self, cx=0.50, cy=0.50):
        """A fist with the thumb deliberately raised past the knuckles."""

        hand = make_hand(CURLED, CURLED, CURLED, CURLED, cx=cx, cy=cy)
        hand[1] = Point(cx - 0.075, cy + 0.120)
        hand[2] = Point(cx - 0.090, cy + 0.050)
        hand[3] = Point(cx - 0.100, cy - 0.010)
        hand[4] = Point(cx - 0.105, cy - 0.055)
        return hand

    def tucked(self, cx=0.50, cy=0.50):
        """Four straight fingers with the thumb across the palm."""

        hand = make_hand(cx=cx, cy=cy)
        hand[2] = Point(cx - 0.075, cy + 0.100)
        hand[3] = Point(cx - 0.050, cy + 0.050)
        hand[4] = Point(cx - 0.030, cy + 0.010)
        return hand

    def test_a_thumbs_up_is_not_a_fist(self):
        self.assertNotEqual(self.settle(self.thumbs_up()), "FIST")

    def test_a_true_fist_still_is(self):
        self.assertEqual(
            self.settle(make_hand(CURLED, CURLED, CURLED, CURLED)), "FIST")

    def test_four_fingers_with_the_thumb_tucked_is_not_a_palm(self):
        self.assertNotEqual(self.settle(self.tucked()), "OPEN_PALM")

    def test_a_whole_open_hand_still_is(self):
        self.assertEqual(self.settle(make_hand()), "OPEN_PALM")


class TestTwoHandsAreTheirOwnVocabulary(GestureTestCase):
    """Both hands making a pose is a different gesture from one hand.

    Raising both palms must not fire what a single palm means: the
    pair reads as 2_OPEN_PALM (and both fists as 2_FIST), names of
    their own, mapped to nothing until someone maps them.  A second
    hand resting in frame suppresses nothing, and one hand alone
    behaves exactly as it always did.
    """

    def palm(self, cx=0.50):
        return make_hand(cx=cx)

    def fist(self, cx=0.50):
        return make_hand(CURLED, CURLED, CURLED, CURLED, cx=cx)

    def paired(self, hand, partner, frames=6):
        """Settle the primary while the partner is in frame."""

        result = "UNKNOWN"
        for _ in range(frames):
            settled = gestures.detect_gesture(hand, HAND)
            result = gestures.pair(settled, partner)
        return result

    def test_both_palms_read_as_the_two_hand_palm(self):
        self.assertEqual(self.paired(self.palm(0.35), self.palm(0.65)),
                         "2_OPEN_PALM")

    def test_both_fists_read_as_the_two_hand_fist(self):
        self.assertEqual(self.paired(self.fist(0.35), self.fist(0.65)),
                         "2_FIST")

    def test_both_points_read_as_the_two_hand_point(self):
        point = lambda cx: make_hand(EXTENDED, CURLED, CURLED, CURLED,
                                     cx=cx)

        self.assertEqual(self.paired(point(0.35), point(0.65)), "2_POINT")

    def test_both_peace_signs_read_as_the_two_hand_two_finger(self):
        self.assertEqual(self.paired(peace_sign(0.35), peace_sign(0.65)),
                         "2_TWO_FINGER")

    def test_two_different_poses_fire_nothing(self):
        self.assertEqual(self.paired(self.palm(0.35), self.fist(0.65)),
                         "UNKNOWN")

    def test_a_resting_second_hand_suppresses_nothing(self):
        rest = make_hand(LOOSE, LOOSE, LOOSE, LOOSE, cx=0.65)

        self.assertEqual(self.paired(self.palm(0.35), rest), "OPEN_PALM")

    def test_one_hand_alone_is_untouched(self):
        self.assertEqual(self.paired(self.palm(), None), "OPEN_PALM")

    def test_the_partner_needs_a_streak_before_it_is_believed(self):
        """One frame of a misread partner must not rename a held pose."""

        settled = "UNKNOWN"
        for _ in range(5):
            settled = gestures.detect_gesture(self.palm(0.35), HAND)

        self.assertEqual(gestures.pair(settled, self.palm(0.65)),
                         "OPEN_PALM")

    def test_reading_the_partner_never_disturbs_the_primary(self):
        """classify_still must stay stateless: reading the second hand
        through the shared finger memories would poison the first."""

        for _ in range(5):
            settled = gestures.detect_gesture(self.palm(0.35), HAND)
            gestures.classify_still(self.fist(0.65))

        self.assertEqual(settled, "OPEN_PALM")


if __name__ == "__main__":
    unittest.main(verbosity=2)
