"""Held gestures: what shape the hand is making right now.

    detect_gesture(landmarks, handedness) -> "OPEN_PALM" | "FIST" |
                                             "POINT" | "TWO_FINGER" |
                                             "UNKNOWN"

Which way the hand faces is not checked.  It used to be, and that quietly
ruled out the two most natural ways to make these: a punch shows the
camera its knuckles rather than the palm, and pointing at the lens turns
the hand edge-on.  Both were rejected before their fingers were ever
counted.
"""

from __future__ import annotations

from . import hand_state
from .state_machine import GestureStabiliser

POSE_GUN = "gun"      # two fingers aimed at the camera
POSE_PEACE = "peace"  # two fingers held up, palm toward the camera
POSE_PINCH = "pinch"  # thumb and index finger together

_stabiliser = GestureStabiliser()


def reset():
    _stabiliser.clear()


def classify(hand, handedness=None):
    """The shape this single frame shows, before stabilising."""

    screen = hand_state.screen_of(hand)
    handedness = getattr(hand, "handedness", handedness)

    # A fist and an open hand are not read from behind: the back of a hand
    # is what the camera sees of someone typing or resting their hand on
    # the desk, and that should ask for nothing.  Side-on is fine -- a
    # fist from the side is still a fist -- so what is ruled out is
    # looking at the back of it, rather than anything short of a square
    # palm.  The two-finger poses are exempt entirely: the swipe can be
    # made with the fingers aimed at the lens, where the palm faces
    # neither way.
    from_behind = hand_state.is_back_of_hand(screen, handedness)

    # The fist first.  A shut hand lays the thumb against the curled index
    # finger, which looks much like a pinch, so whichever is asked second
    # never gets a look at a genuine one of the other.  A fist is the less
    # ambiguous of the two -- every finger shut -- so it answers first, and
    # a real pinch is unaffected because its index finger is out.
    if hand_state.is_clenched(screen):
        return "UNKNOWN" if from_behind else "FIST"

    if hand_state.is_pinching(hand):
        return "PINCH"

    fingers = hand_state.fingers_out(hand)
    extended = sum(fingers.values())

    if extended == 0:
        # Not shut, but nothing straight either: a hand at rest.
        return "UNKNOWN"

    if (
        fingers["index"]
        and not fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    ):
        return "POINT"

    if (
        fingers["index"]
        and fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    ):
        return "TWO_FINGER"

    if extended == 4:
        # Properly open, not merely not-closed: a hand at rest has fingers
        # straighter than a fist and slacker than a spread hand, and it was
        # landing here.
        if from_behind or not hand_state.is_open(hand_state.shape_of(hand)):
            return "UNKNOWN"

        return "OPEN_PALM"

    return "UNKNOWN"


def detect_gesture(hand, handedness=None):
    """The settled gesture, once it has held for a few frames.

    ``handedness`` is needed to tell a palm from the back of a hand: the
    two are mirror images, so which hand it is decides which is which.
    It is read from the hand itself when there is one.
    """

    return _stabiliser.update(classify(hand, handedness))


def pose_kind(hand):
    """Which pose a movement could be made from, or None.

    Two fingers for the sideways turn, a pinch for up and down.  Keeping
    them apart means a hand raised while seeking cannot be read as volume,
    and the two gestures no longer have to be told apart by direction
    alone.
    """

    if hand_state.is_pinching(hand):
        return POSE_PINCH

    return two_finger_pose_kind(hand)


def two_finger_pose_kind(hand):
    """Which two-finger pose this is, or None.

    Both are index and middle out with ring and pinky in; they differ only
    in which way the hand faces, and therefore in how you swipe with them.
    Establishing the fingers first and the direction second matters: the
    finger test holds at any angle, so the pose is never missed because
    the hand happened to be turned.
    """

    fingers = hand_state.fingers_out(hand)

    two_out = (
        fingers["index"]
        and fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    )

    if not two_out:
        return None

    if hand_state.fingers_aimed_at_camera(hand_state.shape_of(hand)):
        return POSE_GUN

    return POSE_PEACE


def is_two_finger_pose(hand):
    """Index and middle out, ring and pinky in -- in either orientation."""

    return two_finger_pose_kind(hand) is not None
