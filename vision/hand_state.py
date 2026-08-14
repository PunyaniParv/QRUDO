"""What a hand is doing, measured from its landmarks.

Everything here answers a question about a single frame -- how straight a
finger is, which way it points, how big the hand looks.  Nothing here
remembers anything; that is state_machine.py's job.

The measurements are deliberately built from 3D distances and directions
rather than from how the hand projects onto the screen.  Turning a hand
side-on changes every screen measurement -- the knuckle row collapses to
almost nothing -- while the hand itself has not changed at all.  Anything
scaled by the screen becomes meaningless at that angle, which is what used
to make a fist in profile go unrecognised.
"""

from __future__ import annotations

import math

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
# ---------------------------------------------------------

#: A finger is out when the straight line from its knuckle to its tip is
#: nearly as long as the finger itself.  Curling a finger shortens that
#: line; turning the hand does not.
EXTENDED_RATIO = 0.82

#: How squarely the fingers must aim at the camera to count as the gun
#: pose rather than the peace sign.  1.0 is straight down the lens, 0.5 is
#: sixty degrees off it.
AIM_AT_CAMERA = 0.50

#: A palm seen edge-on gives a normal with almost no depth component, and
#: its sign is then decided by noise.  Below this the answer is "cannot
#: tell" rather than a coin flip.
PALM_CERTAINTY = 0.12


# ---------------------------------------------------------
# Geometry
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
    the hand turning is scaled by it.

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
    zero means edge-on.  Nothing in the detectors depends on this any
    more: it is decisive for a hand held flat and meaningless otherwise,
    and putting it in front of the other tests is what once stopped a
    punch, a pointed finger and every swipe from registering.
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
# Fingers
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


def finger_scores(hand_landmarks):
    """Every finger's extension score, for the tuning overlay."""

    return {
        name: round(finger_extension(hand_landmarks, tip, pip, mcp), 2)
        for name, (tip, pip, mcp) in FINGERS.items()
    }


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
    """Whether the index and middle fingers point at the lens."""

    return (
        finger_aim(hand_landmarks, 8, 5) > AIM_AT_CAMERA
        and finger_aim(hand_landmarks, 12, 9) > AIM_AT_CAMERA
    )


def pointing_direction(hand_landmarks):
    """Which way the two fingers point: -1 left, 0 at the camera, +1 right.

    The sideways part of the wrist-to-fingertip vector divided by that
    vector's full 3D length, so it measures how far the hand has turned
    and nothing else.

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
