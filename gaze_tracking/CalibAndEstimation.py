import cv2
import numpy as np
from gaze_traking2 import GazeTrackingMediaPipe
from collections import deque
import pyautogui


class EnhancedGazeTracker:
    def __init__(self):
        # Initialize gaze, camera, and screen parameters
        self.gaze = GazeTrackingMediaPipe()
        self.webcam = cv2.VideoCapture(0)
        # self.webcam.set(cv2.CAP_PROP_FPS, 30)
        # self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.coefs = []
        self.screen_width, self.screen_height = 1920, 1080 #2560, 1440

        # Calibration params
        xs = np.linspace(0.05, 0.95, 4)
        ys = np.linspace(0.05, 0.95, 5)
        self.calibration_points = [(x, y) for y in ys for x in xs]
        
        self.calibration_data = []
        self.current_calibration_index = 0
        self.calibrating = True
        self.x_coeff = None
        self.y_coeff = None

        # Smoothing
        self.history = deque(maxlen=5)

    def perform_calibration(self, frame):
        point = self.calibration_points[self.current_calibration_index]
        x = int(point[0] * frame.shape[1])
        y = int(point[1] * frame.shape[0])

        # Draw calibration point
        cv2.circle(frame, (x, y), 8, (0, 255, 0), 2)
        cv2.putText(frame, f"Look at the circle ({self.current_calibration_index + 1}/{len(self.calibration_points)})",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE when ready", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def calculate_mapping(self):
        """Calculate polynomial mapping from gaze to screen coordinates"""
        gaze_x = [d['gaze'][0] for d in self.calibration_data]
        gaze_y = [d['gaze'][1] for d in self.calibration_data]
        screen_x = [d['screen'][0] for d in self.calibration_data]
        screen_y = [d['screen'][1] for d in self.calibration_data]

        # Fit 2nd degree polynomial
        self.x_coeff = np.polyfit(gaze_x, screen_x, 2)
        self.y_coeff = np.polyfit(gaze_y, screen_y, 2)
        self.coefs = [self.x_coeff, self.y_coeff]
        print(self.coefs)
        np.save('calib_data.npz', np.array(self.coefs))
        self.calibration_complete = True

    def smooth_gaze(self, h_ratio, v_ratio):
        """Apply smoothing to reduce jitter"""
        self.history.append((h_ratio, v_ratio))
        if len(self.history) == 0:
            return h_ratio, v_ratio

        # Weighted average (recent frames have more weight)
        weights = np.linspace(0.1, 1.0, len(self.history))
        sum_weights = np.sum(weights)

        avg_h = sum(h * w for (h, v), w in zip(self.history, weights)) / sum_weights
        avg_v = sum(v * w for (h, v), w in zip(self.history, weights)) / sum_weights

        return avg_h, avg_v

    def map_to_screen(self, h_ratio, v_ratio):
        """Map gaze ratios to screen coordinates using calibration"""
        if not self.calibration_complete:
            return int(h_ratio * self.screen_width), int(v_ratio * self.screen_height)

        # Apply polynomial mapping
        screen_x = np.polyval(self.x_coeff, h_ratio)
        screen_y = np.polyval(self.y_coeff, v_ratio)

        # Clamp to screen bounds
        screen_x = max(0, min(self.screen_width, int(screen_x * self.screen_width)))
        screen_y = max(0, min(self.screen_height, int(screen_y * self.screen_height)))

        return screen_x, screen_y

    def Calibrun(self):
        cv2.namedWindow("GazeTracker", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("GazeTracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        pyautogui.FAILSAFE = False
        # camera_matrix = np.array([[400, 0, 540],
        #                           [0, 400, 540],
        #                           [0, 0, 1]])
        # dist_coeffs = np.array([[0, 0, 0, 0, 1]])
        # Calibration loop
        while self.calibrating:
            ret, frame = self.webcam.read()
            if not ret:
                break

            # frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            self.gaze.refresh(frame)
            frame = self.gaze.annotated_frame()
            frame = self.perform_calibration(frame)
            cv2.imshow("GazeTracker", frame)
            key = cv2.waitKey(1)
            if key == 27:
                self.webcam.release()
                cv2.destroyAllWindows()
                return
            if key == 32 and self.gaze.pupils_located:
                h = self.gaze.horizontal_ratio()
                v = self.gaze.vertical_ratio()
                if h is not None:
                    self.calibration_data.append({
                        'gaze': (h, v),
                        'screen': self.calibration_points[self.current_calibration_index]
                    })
                    self.current_calibration_index += 1
                    if self.current_calibration_index >= len(self.calibration_points):
                        self.calculate_mapping()
                        break
        if key == ord('r'):
            # restart calibration
            self.__init__()
            return self.run()

        # self.webcam.release()
        cv2.destroyAllWindows()
    def run(self, frame):
        # Main tracking loop
        self.gaze.refresh(frame)
        frame = self.gaze.annotated_frame()

        if self.gaze.pupils_located:
            h = self.gaze.horizontal_ratio()
            v = self.gaze.vertical_ratio()
            if h is not None:
                h_s, v_s = self.smooth_gaze(h, v)
                sx, sy = self.map_to_screen(h_s, v_s)
                # pyautogui.moveTo(sx, sy)
                # cv2.circle(frame, (sx // (self.screen_width // frame.shape[1]), sy // (self.screen_height // frame.shape[0])), 10, (0, 0, 255), -1)

        # cv2.imshow("GazeTracker", frame)
        key = cv2.waitKey(1)

        return sx, sy

# if __name__ == "__main__":
    # tracker = EnhancedGazeTracker()
    # tracker.Calibrun()
    # cv2.namedWindow("sads", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("sads", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # pyautogui.FAILSAFE = False
    # ret, frame = tracker.webcam.read()
    #
    # if ret:
    #     print(tracker.run(frame))
    # else:
    #     print("Error: frame passing error")
