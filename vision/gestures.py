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

_stabiliser = GestureStabiliser()


def reset():
    _stabiliser.clear()


def classify(hand):
    """The shape this single frame shows, before stabilising."""

    fingers = hand_state.detect_fingers(hand_state.shape_of(hand))
    extended = sum(fingers.values())

    if extended == 0:
        return "FIST"

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
        return "OPEN_PALM"

    return "UNKNOWN"


def detect_gesture(hand, handedness=None):
    """The settled gesture, once it has held for a few frames.

    ``handedness`` is accepted and ignored; nothing here depends on which
    hand it is any more.  It is kept so callers do not have to change.
    """

    return _stabiliser.update(classify(hand))


def two_finger_pose_kind(hand):
    """Which two-finger pose this is, or None.

    Both are index and middle out with ring and pinky in; they differ only
    in which way the hand faces, and therefore in how you swipe with them.
    Establishing the fingers first and the direction second matters: the
    finger test holds at any angle, so the pose is never missed because
    the hand happened to be turned.
    """

    shape = hand_state.shape_of(hand)

    fingers = hand_state.detect_fingers(shape)

    two_out = (
        fingers["index"]
        and fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    )

    if not two_out:
        return None

    if hand_state.fingers_aimed_at_camera(shape):
        return POSE_GUN

    return POSE_PEACE


def is_two_finger_pose(hand):
    """Index and middle out, ring and pinky in -- in either orientation."""

    return two_finger_pose_kind(hand) is not None
