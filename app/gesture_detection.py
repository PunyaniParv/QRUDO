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

# A finger is out when the straight line from its knuckle to its tip is
# nearly as long as the finger itself.  Curling a finger shortens that
# line; turning the hand does not.
EXTENDED_RATIO = 0.82

# How squarely the fingers must aim at the camera to count as the gun
# pose rather than the peace sign.  1.0 is straight down the lens, 0.5 is
# sixty degrees off it.
AIM_AT_CAMERA = 0.50

# A palm seen edge-on gives a normal with almost no depth component, and
# its sign is then decided by noise.  Below this the answer is "cannot
# tell" rather than a coin flip.
PALM_CERTAINTY = 0.12

# Swipes: the motion itself.
#
# The gesture is a wrist rotation, not a hand movement: the wrist stays
# put and the two fingers swing left or right like a pointer.  So what is
# measured is which way the fingers aim, from -1 (hard left) through 0
# (straight at the camera) to +1 (hard right).  Turning far enough, fast
# enough, in one direction is a swipe.
SWIPE_WINDOW = 0.60       # seconds of movement considered
SWIPE_TURN = 0.30         # how far the aim must swing
SWIPE_TURN_SPEED = 0.80   # per second
SWIPE_CONSISTENCY = 0.65  # how directly the motion got where it ended up
SWIPE_MIN_SAMPLES = 4
ARM_HOLD = 0.60           # how long the pose keeps a swipe allowed
SWIPE_COOLDOWN = 0.60

# The peace sign is held up rather than aimed, and the natural way to
# swipe it is to move the hand across rather than pivot the wrist.  So
# that pose also accepts a sideways slide, measured in palm widths.
#
# The gun pose deliberately does not: aiming at the camera and carrying
# your hand to the keyboard should not seek the video.
SWIPE_SLIDE = 0.90        # palm widths the hand must cover
SWIPE_SLIDE_SPEED = 1.60  # palm widths per second


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
swipe_armed_kind = None
swipe_cooldown_until = 0.0

# Filled in every frame for the on-screen readout.
_debug = {}


def reset_state():
    """Forget everything.  Useful when the hand leaves the frame."""

    gesture_history.clear()
    motion_history.clear()

    global swipe_armed_until
    global swipe_armed_kind

    swipe_armed_until = 0.0
    swipe_armed_kind = None


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
    """How big the hand looks on screen, for measuring movement across it.

    Only the sideways slide uses this, and a slide is a screen distance,
    so a screen measurement is the right one.  Nothing that has to survive
    the hand turning is scaled by it -- turn the hand side-on and the
    knuckle row collapses to almost nothing, which is what made side
    profile misbehave.

    Both the palm's width and its length are considered, because a turn
    that flattens one leaves the other.
    """

    return max(
        distance_2d(hand_landmarks[INDEX_MCP], hand_landmarks[PINKY_MCP]),
        distance_2d(hand_landmarks[WRIST], hand_landmarks[MIDDLE_MCP]),
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
# Finger state
# ---------------------------------------------------------

def finger_extension(hand_landmarks, tip, pip, mcp):
    """How straight a finger is: about 1.0 straight, about 0.5 curled.

    The straight-line distance from knuckle to tip, divided by the length
    of the finger itself.  Both are measured in 3D, which is what makes
    this hold from any viewpoint: turning the hand changes how long the
    finger *looks*, but not how long it *is*.

    It also shrugs off MediaPipe's shaky depth.  If every z is off by the
    same factor, both distances are off by roughly that factor too, and
    the ratio between them barely moves.

    This replaced joint angles, which are only trustworthy when the hand
    lies flat to the camera -- a finger aimed at the lens collapses into a
    short cluster of landmarks and reads as bent however straight it is.
    """

    dip = pip + 1

    span = distance(hand_landmarks[mcp], hand_landmarks[tip])

    length = (
        distance(hand_landmarks[mcp], hand_landmarks[pip]) +
        distance(hand_landmarks[pip], hand_landmarks[dip]) +
        distance(hand_landmarks[dip], hand_landmarks[tip])
    )

    if length == 0:
        return 0.0

    return span / length


def finger_is_extended(hand_landmarks, tip, pip, mcp):
    """Whether a finger is out, seen from any angle."""

    return finger_extension(
        hand_landmarks, tip, pip, mcp
    ) > EXTENDED_RATIO


def detect_fingers(hand_landmarks):
    """Which fingers are out."""

    return {
        name: finger_is_extended(hand_landmarks, tip, pip, mcp)
        for name, (tip, pip, mcp) in FINGERS.items()
    }


#: Kept as a name: extension no longer depends on which way the hand
#: faces, so there is nothing extra to ask.
detect_fingers_out = detect_fingers


# ---------------------------------------------------------
# Finger state -- fingers toward camera
# ---------------------------------------------------------

def finger_aim(hand_landmarks, tip, mcp):
    """How squarely a finger points at the camera: 1.0 down the lens, 0 across.

    A direction rather than a distance, so it needs no scaling -- which
    matters, because every screen-based scale collapses when the hand
    turns side-on.
    """

    span = distance(hand_landmarks[mcp], hand_landmarks[tip])

    if span == 0:
        return 0.0

    return (hand_landmarks[mcp].z - hand_landmarks[tip].z) / span


def fingers_aimed_at_camera(hand_landmarks):
    """Whether the index and middle fingers point at the lens.

    Only asked once a pose is already known to have those two fingers
    out, so it decides which pose it is, never whether there is one.
    """

    return (
        finger_aim(hand_landmarks, 8, 5) > AIM_AT_CAMERA
        and finger_aim(hand_landmarks, 12, 9) > AIM_AT_CAMERA
    )


# ---------------------------------------------------------
# Where the fingers aim
# ---------------------------------------------------------

def pointing_direction(hand_landmarks):
    """Which way the two fingers point: -1 left, 0 at the camera, +1 right.

    This is the sideways part of the wrist-to-fingertip vector divided by
    that vector's full 3D length, so it measures how far the hand has
    turned and nothing else.

    Two things follow, both wanted.  Sliding the whole hand across the
    frame does not change it, so drifting your arm cannot look like a
    swipe.  And it does not care how far away you sit, because it is a
    ratio rather than a distance.
    """

    wrist = hand_landmarks[WRIST]

    tip_x = (hand_landmarks[8].x + hand_landmarks[12].x) / 2
    tip_y = (hand_landmarks[8].y + hand_landmarks[12].y) / 2
    tip_z = (hand_landmarks[8].z + hand_landmarks[12].z) / 2

    reach = math.sqrt(
        (tip_x - wrist.x) ** 2 +
        (tip_y - wrist.y) ** 2 +
        (tip_z - wrist.z) ** 2
    )

    if reach < 1e-6:
        return 0.0

    return (tip_x - wrist.x) / reach


# ---------------------------------------------------------
# Two-finger pose
# ---------------------------------------------------------

POSE_GUN = "gun"      # two fingers aimed at the camera
POSE_PEACE = "peace"  # two fingers held up, palm toward the camera


def two_finger_pose_kind(hand_landmarks):
    """Which two-finger pose this is, or None.

    Both are index and middle out with ring and pinky in; they differ only
    in which way the hand faces, and therefore in how you swipe with them.
    Establishing the fingers first and the direction second matters: the
    finger test holds at any angle, so the pose is never missed because
    the hand happened to be turned.
    """

    fingers = detect_fingers(hand_landmarks)

    two_out = (
        fingers["index"]
        and fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    )

    if not two_out:
        return None

    if fingers_aimed_at_camera(hand_landmarks):
        return POSE_GUN

    return POSE_PEACE


def is_two_finger_pose(hand_landmarks):
    """Index and middle out, ring and pinky in -- in either orientation."""

    return two_finger_pose_kind(hand_landmarks) is not None


# ---------------------------------------------------------
# Swipe detection
# ---------------------------------------------------------

def _measure(values, elapsed):
    """Summarise one signal over the window: (change, speed, directness).

    Directness compares where the signal ended up against how far it
    travelled getting there.  A clean movement scores near 1; a hand
    shaking between two positions covers ground and arrives nowhere, so it
    scores low however fast it shakes.  Counting which frames moved the
    right way is not enough -- alternating samples can reach two thirds
    agreement by luck, which is exactly how a wobble used to fire a swipe.
    """

    change = values[-1] - values[0]

    steps = [
        values[i + 1] - values[i]
        for i in range(len(values) - 1)
    ]

    path = sum(abs(step) for step in steps)

    directness = abs(change) / path if path > 0 else 0.0

    return change, abs(change) / elapsed, directness

def detect_swipe(hand_landmarks, handedness):
    """Detect a two-finger swipe.

    Returns "SWIPE_LEFT", "SWIPE_RIGHT" or None.

    The gesture is a wrist rotation: hold two fingers toward the camera
    and turn them left or right, keeping the wrist where it is.

    Where the fingers aim is recorded on every frame, whatever the pose;
    the pose only decides whether a swipe is currently allowed.  A frame
    or two of misclassification therefore costs a sample rather than the
    whole gesture -- which matters here, because a hand turning away from
    the camera passes through angles that are awkward to classify.
    """

    global swipe_armed_until
    global swipe_armed_kind
    global swipe_cooldown_until

    now = time.time()
    scale = hand_scale(hand_landmarks)

    motion_history.append((
        now,
        pointing_direction(hand_landmarks),
        hand_landmarks[MIDDLE_MCP].x,
        scale
    ))

    kind = two_finger_pose_kind(hand_landmarks)

    if kind is not None:
        swipe_armed_until = now + ARM_HOLD
        swipe_armed_kind = kind

    armed = now < swipe_armed_until

    _debug.update(
        pose=kind,
        armed=armed,
        aim=round(motion_history[-1][1], 2),
        turn=0.0,
        slide=0.0,
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

    # Turning the wrist: works with either pose.
    turn, turn_speed, turn_agree = _measure(
        [sample[1] for sample in window], elapsed)

    # Moving the hand across: only the peace sign, and measured in palm
    # widths so distance from the camera does not matter.
    mean_scale = sum(sample[3] for sample in window) / len(window)

    slide, slide_speed, slide_agree = _measure(
        [sample[2] / mean_scale for sample in window], elapsed)

    _debug.update(
        turn=round(abs(turn), 2),
        slide=round(abs(slide), 2),
        speed=round(turn_speed, 2),
        agree=round(turn_agree, 2)
    )

    turned = (
        abs(turn) >= SWIPE_TURN
        and turn_speed >= SWIPE_TURN_SPEED
        and turn_agree >= SWIPE_CONSISTENCY
    )

    slid = (
        swipe_armed_kind == POSE_PEACE
        and abs(slide) >= SWIPE_SLIDE
        and slide_speed >= SWIPE_SLIDE_SPEED
        and slide_agree >= SWIPE_CONSISTENCY
    )

    if not turned and not slid:
        return None

    # -------------------------------------------------
    # Direction
    # -------------------------------------------------

    moving_right = (turn if turned else slide) > 0

    if not FRAME_IS_MIRRORED:
        moving_right = not moving_right

    gesture = "SWIPE_RIGHT" if moving_right else "SWIPE_LEFT"

    # Start fresh, so the tail of this swipe cannot trigger another.
    motion_history.clear()
    swipe_armed_until = 0.0
    swipe_armed_kind = None
    swipe_cooldown_until = now + SWIPE_COOLDOWN

    return gesture


# ---------------------------------------------------------
# Static gesture detection
# ---------------------------------------------------------

def detect_gesture(hand_landmarks, handedness):
    """Detect a held gesture: OPEN_PALM, FIST, POINT, TWO_FINGER, UNKNOWN.

    Which way the hand faces is not checked.  It used to be, and that
    quietly ruled out the two most natural ways to make these gestures: a
    punch shows the camera its knuckles rather than the palm, and pointing
    at the lens turns the hand edge-on.  Both were rejected before their
    fingers were ever counted.

    The cost is that the gun pose also reads as TWO_FINGER here, since two
    fingers really are out.  Swipes are reported separately by
    detect_swipe, so nothing is lost unless TWO_FINGER is later bound to a
    command of its own.
    """

    fingers = detect_fingers_out(hand_landmarks)
    extended_count = sum(fingers.values())

    # Per-finger scores for the tuning overlay.  If a gesture is misread at
    # some angle, this says which finger the camera disagrees about and by
    # how much, which is the difference between adjusting one threshold and
    # guessing.
    _debug["ext"] = {
        name: round(finger_extension(hand_landmarks, tip, pip, mcp), 2)
        for name, (tip, pip, mcp) in FINGERS.items()
    }

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
