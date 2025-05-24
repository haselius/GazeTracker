from __future__ import division

import math
import cv2
import os
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp


class Eye(object):
    def __init__(self, frame, eye_landmarks, iris_landmarks):
        self.frame = frame
        self.landmarks = eye_landmarks
        self.iris = iris_landmarks
        self._analyze()

    def _analyze(self):
        # Calculate eye region coordinates
        self.x, self.y = self._get_min_coordinates()
        self.w, self.h = self._get_max_dimensions()
        #shift

        # Calculate pupil position as iris center
        self.pupil = self._calculate_pupil_position()

    def _get_min_coordinates(self):
        return (min(pt[0] for pt in self.landmarks),
                min(pt[1] for pt in self.landmarks))

    def _get_max_dimensions(self):
        #forsing it make it square
        width = max(pt[0] for pt in self.landmarks) - self.x
        height = max(pt[1] for pt in self.landmarks) - self.y
        size = max(width, height)
        return size, size  # force square

    def _calculate_pupil_position(self):
        # Use iris landmarks to find center
        iris_center = (
            sum(pt[0] for pt in self.iris) // len(self.iris),
            sum(pt[1] for pt in self.iris) // len(self.iris)
        )
        return (iris_center[0] - self.x, iris_center[1] - self.y)


class GazeTrackingMediaPipe(object):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.CLOSE_LEFT = [33, 160, 158, 133, 153, 144]
        self.CLOSE_RIGHT = [362, 385, 387, 263, 373, 380]
        # MediaPipe face landmark indices
        self.LEFT_EYE = [33, 133, 246, 161]
        self.RIGHT_EYE = [362, 263, 466, 388]

        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        self.frame = None
        self.eye_left = None
        self.eye_right = None
        # blinking detetction
        self.close_left = 0.0
        self.close_right = 0.0

    def refresh(self, frame):
        self.frame = frame
        self._analyze_face()

    def _analyze_face(self):
        img_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Extract eye regions
            left_eye = self._get_landmark_points(face_landmarks, self.LEFT_EYE)
            right_eye = self._get_landmark_points(face_landmarks, self.RIGHT_EYE)

            # Extract iris points
            left_iris = self._get_landmark_points(face_landmarks, self.LEFT_IRIS)
            right_iris = self._get_landmark_points(face_landmarks, self.RIGHT_IRIS)

            # Extract points for blinking detection
            left_close_points = self._get_landmark_points(face_landmarks, self.CLOSE_LEFT)
            right_close_points = self._get_landmark_points(face_landmarks, self.CLOSE_RIGHT)

            # Create Eye objects
            self.eye_left = Eye(self.frame, left_eye, left_iris)
            self.eye_right = Eye(self.frame, right_eye, right_iris)

            # Calculate Eye closing Ratio
            self.close_left = self._calculate_ratio(left_close_points)
            self.ear_right = self._calculate_ratio(right_close_points)
        else:
            self.eye_left = None
            self.eye_right = None
            self.close_left = 0.0
            self.close_right = 0.0

    def _calculate_ratio(self, eye_points):
        # Calculate vertical distances
        vertical1 = math.dist(eye_points[1], eye_points[5])
        vertical2 = math.dist(eye_points[2], eye_points[4])

        # Calculate horizontal distance
        horizontal = math.dist(eye_points[0], eye_points[3])

        # Compute EAR
        return (vertical1 + vertical2) / (2.0 * horizontal)

    def is_blinking(self, threshold=0.021, use_avg=True):
        """
        Detect blink using height of an eye
        - threshold: threshold for blink detection (default 0.21)
        - use_avg: Use average of both eyes when True, detect per-eye when False for debug purpose
        """
        if use_avg:
            avg_ear = (self.close_left + self.close_right) / 2.0
            return avg_ear < threshold
        else:
            return self.close_left < threshold or self.close_right < threshold

    def _get_landmark_points(self, landmarks, indices):
        h, w = self.frame.shape[:2]
        return [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in indices]

    @property
    def pupils_located(self):
        return self.eye_left is not None and self.eye_right is not None

    def horizontal_ratio(self):
        if self.pupils_located:
            left_pupil = self.eye_left.pupil[0] / self.eye_left.w
            right_pupil = self.eye_right.pupil[0] / self.eye_right.w
            return (left_pupil + right_pupil) / 2
        return 0.5

    def vertical_ratio(self):
        if self.pupils_located:
            left_pupil = self.eye_left.pupil[1] / self.eye_left.h
            right_pupil = self.eye_right.pupil[1] / self.eye_right.h
            return (left_pupil + right_pupil) / 2
        return 0.5

    def annotated_frame(self):
        frame = self.frame.copy()

        if self.pupils_located:
            # Draw eye regions
            for eye in [self.eye_left, self.eye_right]:
                x, y, w, h = eye.x, eye.y, eye.w, eye.h
                cv2.rectangle(frame, (x, y - h//2), (x + w, y + h//2), (0, 255, 0), 1)

                # Draw pupil position
                pupil_x = x + eye.pupil[0]
                pupil_y = y + eye.pupil[1]
                cv2.circle(frame, (pupil_x, pupil_y), 3, (0, 0, 255), -1)

        return frame