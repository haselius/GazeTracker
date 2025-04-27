"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking.gaze_traking2 import GazeTrackingMediaPipe

gaze = GazeTrackingMediaPipe()
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    # Get gaze direction
    if gaze.pupils_located:
        print(f"Horizontal: {gaze.horizontal_ratio():.2f}, Vertical: {gaze.vertical_ratio():.2f}")

    cv2.imshow('Gaze Tracking', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()