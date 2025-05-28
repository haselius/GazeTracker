import cv2
import numpy as np

class KalmanFilterWrapper:
    def __init__(self):
        # 4 dynamic parameters: x, y, dx, dy; 2 measured parameters: x, y
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def correct_and_predict(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            # Initialize state with position [x, y] and zero velocity
            self.kalman.statePre = np.array([[np.float32(x)],
                                             [np.float32(y)],
                                             [0],
                                             [0]], np.float32)
            self.initialized = True

        self.kalman.correct(measurement)
        predicted = self.kalman.predict()
        return predicted[0][0], predicted[1][0]