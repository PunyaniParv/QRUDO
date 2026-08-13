import cv2
import mediapipe as mp

# MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the hand landmark model
MODEL_PATH = "models/hand_landmarker.task"

# Create the hand landmarker
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

landmarker = vision.HandLandmarker.create_from_options(options)

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("SARV hand tracking started.")
print("Press Q to close.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe requires an mp.Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    results = landmarker.detect(mp_image)

    # Draw landmarks
    if results.hand_landmarks:

        for hand_landmarks in results.hand_landmarks:

            height, width, _ = frame.shape

            # Draw each landmark
            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

            for connection in connections:

                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]

                start_point = (
                    int(start.x * width),
                    int(start.y * height)
                )

                end_point = (
                    int(end.x * width),
                    int(end.y * height)
                )

                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("SARV - Hand Tracking", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
landmarker.close()
cv2.destroyAllWindows()