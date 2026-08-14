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

#: A fist is judged by where the fingertips end up rather than by how
#: bent each finger is.  How far a tip sits from the wrist, against how
#: far its knuckle does: shut brings the tips back level with the
#: knuckles, a hand merely resting leaves them well beyond, and an open
#: hand further still.  Those separate cleanly, where "how bent is this
#: finger" does not -- and the knuckles and wrist stay well tracked even
#: when a closed hand hides the fingers themselves.
FIST_REACH = 1.15

#: How squarely the fingers must aim at the camera to count as the gun
#: pose rather than the peace sign.  1.0 is straight down the lens, 0.5 is
#: sixty degrees off it.
AIM_AT_CAMERA = 0.50

#: A palm seen edge-on gives a normal with almost no depth component, and
#: its sign is then decided by noise.  Below this the answer is "cannot
#: tell" rather than a coin flip.
PALM_CERTAINTY = 0.12

#: Smallest a hand can look on screen and still be taken as deliberate,
#: as a fraction of the frame width.  A hand across the room, or one
#: mostly out of shot, is doing something else.
MIN_HAND_ON_SCREEN = 0.09


# ---------------------------------------------------------
# Which landmarks to ask
# ---------------------------------------------------------

def shape_of(hand):
    """Landmarks for measuring what the hand is doing.

    MediaPipe returns two sets and they are not interchangeable.  The
    normalised set says where the hand is on screen, and its depth is only
    loosely scaled -- but a finger aimed at the lens has nearly all of its
    length in that depth, so measured there it comes out short, which reads
    as curled.  That is why the gun pose was mistaken for a fist and swipes
    never armed.

    The world set is in metres with real depth, which is what a question
    about shape needs.  Anything about position on screen must still use
    the normalised set: see screen_of.
    """

    world = getattr(hand, "world", None)

    if world:
        return world

    # A plain list of landmarks, as the tests provide.
    return getattr(hand, "landmarks", hand)


def screen_of(hand):
    """Landmarks for measuring where the hand is in the frame."""

    return getattr(hand, "landmarks", hand)


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


def is_prominent(hand):
    """Whether the hand is close enough and whole enough to mean it."""

    return hand_scale(screen_of(hand)) >= MIN_HAND_ON_SCREEN


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


def is_back_of_hand(hand_landmarks, handedness):
    """True only when the camera is clearly behind the hand.

    Separate from "is the palm facing us", and the useful one of the two.
    A hand turned side-on is neither -- and a fist from the side is still
    a fist -- so the test that matters is whether we are looking at the
    back of it, which is what a hand on a keyboard shows.
    """

    return palm_facing_strength(
        hand_landmarks,
        handedness
    ) < -PALM_CERTAINTY


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


def finger_reach(hand_landmarks, tip, mcp):
    """How far past its knuckle a fingertip sits, as a multiple.

    About 1 when the hand is shut, around 1.3 when it is merely resting,
    and 1.6 or more when the finger is out.
    """

    knuckle = distance(hand_landmarks[WRIST], hand_landmarks[mcp])

    if knuckle == 0:
        return 0.0

    return distance(hand_landmarks[WRIST], hand_landmarks[tip]) / knuckle


def detect_fingers(hand_landmarks):
    """Which fingers are out."""

    return {
        name: finger_is_extended(hand_landmarks, tip, pip, mcp)
        for name, (tip, pip, mcp) in FINGERS.items()
    }


def fingers_out(hand):
    """Which fingers are out, asking both views of the hand.

    A finger counts if either says so.  The screen view is exact for a
    hand held flat and hopeless for one aimed at the camera; the world
    view is the reverse.  Requiring both to agree loses real gestures --
    which is what stopped two fingers registering at all.
    """

    shape = detect_fingers(shape_of(hand))
    screen = detect_fingers(screen_of(hand))

    return {name: shape[name] or screen[name] for name in FINGERS}


def is_clenched(hand_landmarks):
    """Whether the hand is shut, not merely un-straight."""

    return all(
        finger_reach(hand_landmarks, tip, mcp) < FIST_REACH
        for tip, _, mcp in FINGERS.values()
    )


def reach_scores(hand_landmarks):
    """Every finger's reach, for the tuning overlay."""

    return {
        name: round(finger_reach(hand_landmarks, tip, mcp), 2)
        for name, (tip, _, mcp) in FINGERS.items()
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


def pointing_direction(hand):
    """Which way the two fingers lean: -1 hard left, 0 upright, +1 hard right.

    Measured purely on screen, in x and y, with no depth at all.  The
    gesture is a wrist rotation with the fingers staying toward the camera,
    so the whole of it happens in the plane of the image -- which is the
    part MediaPipe reports accurately.  Bringing depth into it would put
    the measurement back on the axis a single camera can only estimate,
    which is what made the earlier version unreliable.

    It stays relative to the wrist, so carrying the hand across the frame
    without turning it reads as nothing, and it is a ratio, so how far away
    you sit does not matter.
    """

    screen = screen_of(hand)

    wrist = screen[WRIST]

    tip_x = (screen[8].x + screen[12].x) / 2
    tip_y = (screen[8].y + screen[12].y) / 2

    sideways = tip_x - wrist.x
    reach = math.hypot(sideways, tip_y - wrist.y)

    # Fingers aimed at the camera collapse toward the wrist on screen, so
    # the reach shrinks and dividing by it would make small movements read
    # as huge -- enough that slowly lowering the hand counted as a swipe.
    # A hand and a half is the floor: comfortably under the reach of an
    # upright peace sign, so that gesture is measured as it stands.
    return sideways / max(reach, hand_scale(screen) * 1.5)
