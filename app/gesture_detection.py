from collections import Counter, deque
import math
import time



gesture_history = deque(maxlen=3)

# Swipe tracking
swipe_start_x = None
swipe_start_time = None
swipe_cooldown_until = 0

def distance(point1, point2):
    """Calculate 3D distance between two landmarks."""
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )


def is_palm_facing(hand_landmarks, handedness):
    """
    Check whether the palm is facing the camera.
    """

    wrist = hand_landmarks[0]
    index_mcp = hand_landmarks[5]
    pinky_mcp = hand_landmarks[17]

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

    normal_z = (
        v1[0] * v2[1]
        - v1[1] * v2[0]
    )

    if handedness == "Right":
        return normal_z < 0

    return normal_z > 0


def calculate_angle(a, b, c):
    """
    Calculate the angle ABC in degrees.
    """

    ba = (
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    )

    bc = (
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    )

    dot_product = (
        ba[0] * bc[0]
        + ba[1] * bc[1]
        + ba[2] * bc[2]
    )

    magnitude_ba = math.sqrt(
        ba[0] ** 2 +
        ba[1] ** 2 +
        ba[2] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2 +
        bc[1] ** 2 +
        bc[2] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine = dot_product / (
        magnitude_ba * magnitude_bc
    )

    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(
        math.acos(cosine)
    )


def finger_is_extended(hand_landmarks, tip, pip, mcp):
    """
    Determine whether a finger is extended
    using joint angles.
    """

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

    return (
        pip_angle > 155
        and dip_angle > 150
    )


def detect_fingers(hand_landmarks):
    """
    Return the extended state of the four fingers.
    """

    return {
        "index": finger_is_extended(hand_landmarks, 8, 6, 5),
        "middle": finger_is_extended(hand_landmarks, 12, 10, 9),
        "ring": finger_is_extended(hand_landmarks, 16, 14, 13),
        "pinky": finger_is_extended(hand_landmarks, 20, 18, 17)
    }


def detect_swipe(hand_landmarks, handedness):
    """
    Detect a horizontal two-finger swipe.

    Returns:
        SWIPE_LEFT
        SWIPE_RIGHT
        None
    """

    global swipe_start_x
    global swipe_start_time
    global swipe_cooldown_until

    current_time = time.time()

    # Still inside cooldown
    if current_time < swipe_cooldown_until:
        return None

    # Palm must face camera
    if not is_palm_facing(hand_landmarks, handedness):
        swipe_start_x = None
        swipe_start_time = None
        return None

    fingers = detect_fingers(hand_landmarks)

    # Require index + middle only
    two_finger_pose = (
        fingers["index"]
        and fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    )

    if not two_finger_pose:
        swipe_start_x = None
        swipe_start_time = None
        return None

    # Midpoint between index and middle fingertips
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]

    current_x = (index_tip.x + middle_tip.x) / 2

    # Start tracking
    if swipe_start_x is None:
        swipe_start_x = current_x
        swipe_start_time = current_time
        return None

    distance_x = current_x - swipe_start_x
    elapsed = current_time - swipe_start_time

    # Reset if movement takes too long
    if elapsed > 0.7:
        swipe_start_x = current_x
        swipe_start_time = current_time
        return None

    # Minimum horizontal movement
    threshold = 0.18

    if abs(distance_x) >= threshold:

        if distance_x > 0:
            gesture = "SWIPE_RIGHT"
        else:
            gesture = "SWIPE_LEFT"

        # Prevent repeated triggers
        swipe_cooldown_until = current_time + 0.8

        swipe_start_x = None
        swipe_start_time = None

        return gesture

    return None


def detect_gesture(hand_landmarks, handedness):
    """
    Detect:
        OPEN_PALM
        FIST
        POINT
        TWO_FINGER
        UNKNOWN

    Swipe direction will be handled separately.
    """

    # Palm must face camera
    if not is_palm_facing(hand_landmarks, handedness):
        gesture_history.clear()
        return "UNKNOWN"

    fingers = detect_fingers(hand_landmarks)

    extended_count = sum(fingers.values())

    # FIST
    if extended_count == 0:
        raw_gesture = "FIST"

    # POINT
    elif (
        fingers["index"]
        and not fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    ):
        raw_gesture = "POINT"

    # TWO FINGER
    elif (
        fingers["index"]
        and fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    ):
        raw_gesture = "TWO_FINGER"

    # OPEN PALM
    elif extended_count == 4:
        raw_gesture = "OPEN_PALM"

    else:
        raw_gesture = "UNKNOWN"

    # Stabilize static gesture
    gesture_history.append(raw_gesture)

    if len(gesture_history) < 3:
        return "UNKNOWN"

    gesture, count = Counter(gesture_history).most_common(1)[0]

    if count >= 2:
        return gesture

    return "UNKNOWN"