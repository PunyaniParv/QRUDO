"""
Gesture detection for SARV.

Public API (unchanged):

    detect_gesture(hand_landmarks, handedness) -> "OPEN_PALM" | "FIST" |
                                                  "POINT" | "TWO_FINGER" |
                                                  "UNKNOWN"
    detect_swipe(hand_landmarks, handedness)   -> "SWIPE_LEFT" |
                                                  "SWIPE_RIGHT" | None

Two hand orientations are supported, because they need different things
from the camera:

  * Static gestures are held with the PALM TOWARD the camera.  The hand
    lies in the image plane, which is where MediaPipe is most accurate,
    so a finger counts as extended when its joints are straight.

  * Swipes are made with two fingers POINTING AT the camera.  In that pose
    the joints are edge-on and foreshortened, so joint angles say very
    little and the palm-facing test says nothing at all -- the palm is
    perpendicular to the lens, so its normal has no depth component to
    test the sign of.  Extension is judged by depth order instead: a
    finger aimed at the lens has its tip nearer the camera than its
    knuckle, while a curled finger tucks its tip back toward the palm.

Both scripts mirror the frame before detection, so x increases to the
user's right and a rightward swipe is a rise in x.  Set FRAME_IS_MIRRORED
to False if that ever changes.
"""

from collections import Counter, deque
import math
import time


# ---------------------------------------------------------
# Landmarks
# ---------------------------------------------------------

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

# name -> (tip, pip, mcp)
FINGERS = {
    "index": (8, 6, 5),
    "middle": (12, 10, 9),
    "ring": (16, 14, 13),
    "pinky": (20, 18, 17),
}


# ---------------------------------------------------------
# Tuning
#
# Every threshold that depends on the hand is expressed in
# palm widths, so sitting closer to or further from the
# camera does not change the behaviour.
# ---------------------------------------------------------

FRAME_IS_MIRRORED = True

# Static gestures: joint angles for a straight finger.  Loosened from
# 155/150, which only held for a perfectly still hand.
PIP_STRAIGHT = 148
DIP_STRAIGHT = 142

# A palm seen edge-on gives a normal with almost no depth component, and
# its sign is then decided by noise.  Below this the answer is "cannot
# tell" rather than a coin flip.
PALM_CERTAINTY = 0.12

# Swipes: how much nearer the camera a fingertip must be than its knuckle
# before the finger counts as pointing at the lens.
POINT_DEPTH = 0.30
CURLED_DEPTH = 0.15

# Swipes: the motion itself.
SWIPE_WINDOW = 0.50       # seconds of movement considered
SWIPE_TRAVEL = 0.90       # palm widths the hand must cover
SWIPE_SPEED = 1.60        # palm widths per second
SWIPE_CONSISTENCY = 0.65  # fraction of frames moving the same way
SWIPE_MIN_SAMPLES = 4
ARM_HOLD = 0.35           # how long the pose keeps a swipe allowed
SWIPE_COOLDOWN = 0.60


# ---------------------------------------------------------
# State
# ---------------------------------------------------------

gesture_history = deque(maxlen=5)

# (timestamp, x, palm width).  Filled on every frame a hand is visible,
# whatever the pose -- the pose decides whether a swipe is allowed, this
# decides whether one happened.  Keeping them separate is what stops a
# single badly classified frame from erasing the whole gesture.
motion_history = deque(maxlen=64)

swipe_armed_until = 0.0
swipe_cooldown_until = 0.0

# Filled in every frame for the on-screen readout.
_debug = {}


def reset_state():
    """Forget everything.  Useful when the hand leaves the frame."""

    gesture_history.clear()
    motion_history.clear()

    global swipe_armed_until
    swipe_armed_until = 0.0


def debug_state():
    """Live values for the tuning overlay."""

    return dict(_debug)


# ---------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------

def distance(point1, point2):
    """3D distance between two landmarks."""

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )


def distance_2d(point1, point2):
    """Distance as it appears on screen, ignoring depth."""

    return math.hypot(
        point1.x - point2.x,
        point1.y - point2.y
    )


def calculate_angle(a, b, c):
    """Angle ABC in degrees."""

    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)

    dot_product = (
        ba[0] * bc[0] +
        ba[1] * bc[1] +
        ba[2] * bc[2]
    )

    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine = dot_product / (magnitude_ba * magnitude_bc)

    # Prevent floating-point errors
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def hand_scale(hand_landmarks):
    """Palm width on screen, used to make thresholds distance-independent.

    The knuckle row is the one measurement that survives both poses: when
    the fingers point at the camera the hand is foreshortened along its
    length, but its width still faces the lens.
    """

    return max(
        distance_2d(
            hand_landmarks[INDEX_MCP],
            hand_landmarks[PINKY_MCP]
        ),
        0.02
    )


# ---------------------------------------------------------
# Palm orientation
# ---------------------------------------------------------

def palm_facing_strength(hand_landmarks, handedness):
    """How squarely the palm faces the camera, as a signed number.

    Positive means facing, negative means the back of the hand, and near
    zero means edge-on -- which is exactly the swipe pose, and why swipes
    must not depend on this.
    """

    wrist = hand_landmarks[WRIST]
    index_mcp = hand_landmarks[INDEX_MCP]
    pinky_mcp = hand_landmarks[PINKY_MCP]

    v1 = (index_mcp.x - wrist.x, index_mcp.y - wrist.y)
    v2 = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y)

    normal_z = v1[0] * v2[1] - v1[1] * v2[0]

    # Scale out hand size so the number means the same at any distance.
    normal_z /= hand_scale(hand_landmarks) ** 2

    if handedness == "Right":
        return -normal_z

    return normal_z


def is_palm_facing(hand_landmarks, handedness):
    """True only when the palm clearly faces the camera."""

    return palm_facing_strength(
        hand_landmarks,
        handedness
    ) > PALM_CERTAINTY


# ---------------------------------------------------------
# Finger state -- palm toward camera
# ---------------------------------------------------------

def finger_is_extended(hand_landmarks, tip, pip, mcp):
    """Straight-finger test, for a hand held flat to the camera."""

    dip = pip + 1

    pip_angle = calculate_angle(
        hand_landmarks[mcp],
        hand_landmarks[pip],
        hand_landmarks[dip]
    )

    dip_angle = calculate_angle(
        hand_landmarks[pip],
        hand_landmarks[dip],
        hand_landmarks[tip]
    )

    return pip_angle > PIP_STRAIGHT and dip_angle > DIP_STRAIGHT


def detect_fingers(hand_landmarks):
    """Which fingers are extended, for a hand held flat to the camera."""

    return {
        name: finger_is_extended(hand_landmarks, tip, pip, mcp)
        for name, (tip, pip, mcp) in FINGERS.items()
    }


# ---------------------------------------------------------
# Finger state -- fingers toward camera
# ---------------------------------------------------------

def finger_points_at_camera(hand_landmarks, tip, pip, mcp, scale):
    """Depth-order test, for a finger aimed at the lens.

    MediaPipe's z grows with distance from the camera, so a finger
    pointing at it has tip nearer than knuckle.  A curled finger folds
    back toward the palm and shows no such gap.
    """

    depth_gap = hand_landmarks[mcp].z - hand_landmarks[tip].z

    # The middle joint should sit between the two, which rules out a
    # fingertip that is only near the camera because the whole hand is
    # tilted.
    ordered = hand_landmarks[pip].z >= hand_landmarks[tip].z

    return depth_gap > POINT_DEPTH * scale and ordered


def detect_pointing_fingers(hand_landmarks):
    """Which fingers are aimed at the camera."""

    scale = hand_scale(hand_landmarks)

    return {
        name: finger_points_at_camera(hand_landmarks, tip, pip, mcp, scale)
        for name, (tip, pip, mcp) in FINGERS.items()
    }


def finger_is_curled(hand_landmarks, tip, mcp, scale):
    """True when a finger is clearly not aimed at the camera."""

    depth_gap = hand_landmarks[mcp].z - hand_landmarks[tip].z

    return depth_gap < CURLED_DEPTH * scale


# ---------------------------------------------------------
# Two-finger pose
# ---------------------------------------------------------

def is_two_finger_pose(hand_landmarks):
    """Index and middle out, ring and pinky in -- in either orientation.

    Accepting both means the swipe still works if the hand drifts from
    pointing at the camera toward facing it, which it does naturally as
    the arm moves sideways.
    """

    scale = hand_scale(hand_landmarks)

    pointing = detect_pointing_fingers(hand_landmarks)

    if (
        pointing["index"]
        and pointing["middle"]
        and finger_is_curled(hand_landmarks, 16, 13, scale)
        and finger_is_curled(hand_landmarks, 20, 17, scale)
    ):
        return True

    flat = detect_fingers(hand_landmarks)

    return (
        flat["index"]
        and flat["middle"]
        and not flat["ring"]
        and not flat["pinky"]
    )


# ---------------------------------------------------------
# Swipe detection
# ---------------------------------------------------------

def detect_swipe(hand_landmarks, handedness):
    """Detect a two-finger swipe.

    Returns "SWIPE_LEFT", "SWIPE_RIGHT" or None.

    The hand's position is recorded on every frame, whatever the pose;
    the pose only decides whether a swipe is currently allowed.  A frame
    or two of misclassification therefore costs a sample rather than the
    whole gesture.
    """

    global swipe_armed_until
    global swipe_cooldown_until

    now = time.time()
    scale = hand_scale(hand_landmarks)

    # Track the middle knuckle rather than the fingertips: it is the
    # centre of the hand, and it does not wander when the fingers are
    # foreshortened.
    motion_history.append((now, hand_landmarks[MIDDLE_MCP].x, scale))

    posed = is_two_finger_pose(hand_landmarks)

    if posed:
        swipe_armed_until = now + ARM_HOLD

    armed = now < swipe_armed_until

    _debug.update(
        pose=posed,
        armed=armed,
        palm=round(palm_facing_strength(hand_landmarks, handedness), 2),
        travel=0.0,
        speed=0.0,
        agree=0.0
    )

    if now < swipe_cooldown_until or not armed:
        return None

    window = [
        sample for sample in motion_history
        if now - sample[0] <= SWIPE_WINDOW
    ]

    if len(window) < SWIPE_MIN_SAMPLES:
        return None

    elapsed = window[-1][0] - window[0][0]

    if elapsed <= 0:
        return None

    # Measure in palm widths so distance from the camera does not matter.
    mean_scale = sum(sample[2] for sample in window) / len(window)
    displacement = (window[-1][1] - window[0][1]) / mean_scale

    travel = abs(displacement)
    speed = travel / elapsed

    # A swipe goes one way.  A wobble does not.
    steps = [
        window[i + 1][1] - window[i][1]
        for i in range(len(window) - 1)
    ]

    forward = sum(
        1 for step in steps
        if step * displacement > 0
    )

    agreement = forward / len(steps) if steps else 0.0

    _debug.update(
        travel=round(travel, 2),
        speed=round(speed, 2),
        agree=round(agreement, 2)
    )

    if travel < SWIPE_TRAVEL:
        return None

    if speed < SWIPE_SPEED:
        return None

    if agreement < SWIPE_CONSISTENCY:
        return None

    # -------------------------------------------------
    # Direction
    # -------------------------------------------------

    moving_right = displacement > 0

    if not FRAME_IS_MIRRORED:
        moving_right = not moving_right

    gesture = "SWIPE_RIGHT" if moving_right else "SWIPE_LEFT"

    # Start fresh, so the tail of this swipe cannot trigger another.
    motion_history.clear()
    swipe_armed_until = 0.0
    swipe_cooldown_until = now + SWIPE_COOLDOWN

    return gesture


# ---------------------------------------------------------
# Static gesture detection
# ---------------------------------------------------------

def detect_gesture(hand_landmarks, handedness):
    """Detect a held gesture: OPEN_PALM, FIST, POINT, TWO_FINGER, UNKNOWN.

    These are meant to be held with the palm toward the camera.  A hand
    that is edge-on -- the swipe pose -- reports UNKNOWN rather than
    guessing, which also stops swipes from firing a static gesture on
    the way past.
    """

    if not is_palm_facing(hand_landmarks, handedness):
        gesture_history.clear()
        return "UNKNOWN"

    fingers = detect_fingers(hand_landmarks)
    extended_count = sum(fingers.values())

    if extended_count == 0:
        raw_gesture = "FIST"

    elif (
        fingers["index"]
        and not fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    ):
        raw_gesture = "POINT"

    elif (
        fingers["index"]
        and fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    ):
        raw_gesture = "TWO_FINGER"

    elif extended_count == 4:
        raw_gesture = "OPEN_PALM"

    else:
        raw_gesture = "UNKNOWN"

    # -------------------------------------------------
    # Stabilise: a gesture must survive a few frames
    # -------------------------------------------------

    gesture_history.append(raw_gesture)

    if len(gesture_history) < gesture_history.maxlen:
        return "UNKNOWN"

    gesture, count = Counter(gesture_history).most_common(1)[0]

    if count >= 3:
        return gesture

    return "UNKNOWN"
