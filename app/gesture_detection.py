from collections import Counter, deque
import math


gesture_history = deque(maxlen=3)


def is_palm_facing(hand_landmarks, handedness):
    """
    Check whether the palm is facing the camera.

    Returns True only when the palm is approximately facing forward.
    """

    wrist = hand_landmarks[0]
    index_mcp = hand_landmarks[5]
    pinky_mcp = hand_landmarks[17]

    # Vectors from wrist to the two sides of the palm
    v1 = (
        index_mcp.x - wrist.x,
        index_mcp.y - wrist.y,
        index_mcp.z - wrist.z
    )

    v2 = (
        pinky_mcp.x - wrist.x,
        pinky_mcp.y - wrist.y,
        pinky_mcp.z - wrist.z
    )

    # Cross product gives the direction the palm is facing
    normal_z = (
        v1[0] * v2[1]
        - v1[1] * v2[0]
    )

    # MediaPipe handedness tells us which hand it is.
    # We use it to determine the expected palm orientation.
    if handedness == "Right":
        return normal_z < 0

    else:
        return normal_z > 0


def detect_gesture(hand_landmarks, handedness):
    """
    Detect basic gestures only when the palm is facing the camera.

    Returns:
        OPEN_PALM
        FIST
        POINT
        UNKNOWN
    """

    # Reject the gesture if the palm isn't facing the camera
    if not is_palm_facing(hand_landmarks, handedness):
        gesture_history.clear()
        return "UNKNOWN"

    fingers = {
        "index": (8, 6),
        "middle": (12, 10),
        "ring": (16, 14),
        "pinky": (20, 18)
    }

    extended = {}

    for finger, (tip, pip) in fingers.items():
        extended[finger] = (
            hand_landmarks[tip].y < hand_landmarks[pip].y
        )

    extended_count = sum(extended.values())

    # Determine raw gesture
    if extended_count == 4:
        raw_gesture = "OPEN_PALM"

    elif extended_count == 0:
        raw_gesture = "FIST"

    elif (
        extended["index"]
        and not extended["middle"]
        and not extended["ring"]
        and not extended["pinky"]
    ):
        raw_gesture = "POINT"

    else:
        raw_gesture = "UNKNOWN"

    # Stabilize the result
    gesture_history.append(raw_gesture)

    if len(gesture_history) < 3:
        return "UNKNOWN"

    gesture, count = Counter(gesture_history).most_common(1)[0]

    if count >= 2:
        return gesture

    return "UNKNOWN"