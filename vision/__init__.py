"""SARV Vision Engine: camera in, gesture names out.

    from vision import Camera, HandTracker, detect_gesture, detect_swipe

Nothing here imports the control side, so gestures can be developed and
tested without touching the machine.  The two halves meet only in
integration/bridge.py.
"""

from .camera import Camera, CameraError
from .gestures import (
    POSE_GUN,
    POSE_PEACE,
    detect_gesture,
    is_two_finger_pose,
    two_finger_pose_kind,
)
from .hand_tracker import Hand, HandTracker, TrackerError
from .state_machine import Presence, now
from .motion import debug_state, detect_swipe

__all__ = [
    "Camera",
    "CameraError",
    "Hand",
    "HandTracker",
    "Presence",
    "now",
    "POSE_GUN",
    "POSE_PEACE",
    "TrackerError",
    "debug_state",
    "detect_gesture",
    "detect_swipe",
    "is_two_finger_pose",
    "reset_state",
    "two_finger_pose_kind",
]


def reset_state():
    """Forget everything remembered about the hand.

    Called when the hand leaves the frame, so a gesture half-made before
    it vanished cannot combine with the next one.
    """

    from . import gestures, motion

    gestures.reset()
    motion.reset()
