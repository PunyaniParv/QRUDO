import sys
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gesture_detection import (
    SWIPE_CONSISTENCY,
    SWIPE_SLIDE,
    SWIPE_TURN,
    SWIPE_TURN_SPEED,
    debug_state,
    detect_gesture,
    detect_swipe,
)


# ---------------------------------------------------------
# Swipe display tracking
# ---------------------------------------------------------

last_swipe = None
last_swipe_time = 0


# ---------------------------------------------------------
# MediaPipe model
# ---------------------------------------------------------

# Relative to the repo, not to wherever you happened to run this from.
MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
)

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ---------------------------------------------------------
# Camera
# ---------------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("SARV gesture detection started.")
print("Press Q to close.")


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    results = landmarker.detect(mp_image)

    # -----------------------------------------------------
    # Hand detected
    # -----------------------------------------------------

    if results.hand_landmarks:

        hand_landmarks = results.hand_landmarks[0]

        handedness = (
            results.handedness[0][0].category_name
        )

        # Static gesture detection
        gesture = detect_gesture(
            hand_landmarks,
            handedness
        )

        # Swipe detection
        swipe = detect_swipe(
            hand_landmarks,
            handedness
        )

        # -------------------------------------------------
        # Store latest swipe event
        # -------------------------------------------------

        if swipe:

            last_swipe = swipe
            last_swipe_time = time.time()

        # -------------------------------------------------
        # Keep swipe visible for 0.7 seconds
        # -------------------------------------------------

        if (
            last_swipe is not None
            and time.time() - last_swipe_time < 0.7
        ):

            gesture = last_swipe

        # -------------------------------------------------
        # Display gesture
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # -------------------------------------------------
        # Live swipe readout
        #
        # Thresholds depend on your camera, your lighting and
        # how far away you sit, so watch these while swiping:
        # whichever line is short of its target is the reason
        # a swipe did not fire.
        # -------------------------------------------------

        state = debug_state()

        readout = [
            f"pose  {state.get('pose')}   armed {state.get('armed')}",
            f"aim   {state.get('aim', 0):+.2f}   (-1 left, +1 right)",
            f"turn  {state.get('turn', 0):.2f} / {SWIPE_TURN}",
            f"slide {state.get('slide', 0):.2f} / {SWIPE_SLIDE}"
            f"   (peace sign only)",
            f"speed {state.get('speed', 0):.2f} / {SWIPE_TURN_SPEED}",
            f"agree {state.get('agree', 0):.2f} / {SWIPE_CONSISTENCY}",
        ]

        for line_number, line in enumerate(readout):

            cv2.putText(
                frame,
                line,
                (20, 100 + line_number * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )

    # -----------------------------------------------------
    # No hand detected
    # -----------------------------------------------------

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # -----------------------------------------------------
    # Show camera
    # -----------------------------------------------------

    cv2.imshow(
        "SARV - Gesture Detection",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

camera.release()
landmarker.close()
cv2.destroyAllWindows()