import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("SARV camera started.")
print("Press Q to close.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    cv2.imshow("SARV - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()