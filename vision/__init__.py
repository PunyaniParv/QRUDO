"""SARV Vision Engine: camera in, gesture names out.

    from vision import Camera, HandTracker, detect_gesture, detect_swipe

Nothing here imports the control side, so gestures can be developed and
tested without touching the machine.  The two halves meet only in
integration/bridge.py.
"""

from .calibration import Calibration, load_and_apply
from .camera import Camera, CameraError
from .gestures import POSE_PINCH, POSE_TWO_FINGER, detect_gesture, pose_kind
from .hand_tracker import Hand, HandTracker, TrackerError
from .state_machine import Presence, now
from .motion import debug_state, detect_swipe

__all__ = [
    "Calibration",
    "Camera",
    "CameraError",
    "Hand",
    "HandTracker",
    "Presence",
    "now",
    "POSE_PINCH",
    "POSE_TWO_FINGER",
    "TrackerError",
    "debug_state",
    "detect_gesture",
    "detect_swipe",
    "load_and_apply",
    "reset_state",
    "pose_kind",
]


def reset_state():
    """Forget everything remembered about the hand.

    Called when the hand leaves the frame, so a gesture half-made before
    it vanished cannot combine with the next one.
    """

    from . import gestures, motion

    gestures.reset()
    motion.reset()
