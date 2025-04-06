

import cv2
import numpy as np
from gaze_traking2 import GazeTrackingMediaPipe
from collections import deque
import pyautogui
import time

class EnhancedGazeTracker:
    def __init__(self):
        self.gaze = GazeTrackingMediaPipe()
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FPS, 30)  # Set webcam frame rate to 30 FPS
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Reduce resolution for faster processing
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.screen_width, self.screen_height = 2560, 1440  # Adjust to your display
        self.pause_play_zone = {
            "h_range": (0.125, 0.875),  # Horizontal range (12.5% to 87.5% of the screen width)
            "v_range": (0.125, 0.875)   # Vertical range (12.5% to 87.5% of the screen height)
        }
        self.last_blink_time = 0  # To track the time of the last blink
        self.blink_count = 0  # To count consecutive blinks
        self.is_playing = True  # Track the current state (playing or paused)

        # Calibration settings
        self.calibration_points = [
            (0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
            (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
            (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)
        ]
        self.calibration_data = []
        self.current_calibration_point = 0
        self.calibrating = True

        # Smoothing
        self.gaze_history = deque(maxlen=5)
        self.calibration_complete = False

        # Mapping coefficients
        self.x_coeff = None
        self.y_coeff = None

    def perform_calibration(self, frame):
        point = self.calibration_points[self.current_calibration_point]
        x = int(point[0] * frame.shape[1])
        y = int(point[1] * frame.shape[0])

        # Draw calibration point
        cv2.circle(frame, (x, y), 15, (0, 255, 0), 2)
        cv2.putText(frame, f"Look at the circle ({self.current_calibration_point+1}/{len(self.calibration_points)})",
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
        self.calibration_complete = True

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

    def smooth_gaze(self, h_ratio, v_ratio):
        """Apply smoothing to reduce jitter"""
        self.gaze_history.append((h_ratio, v_ratio))
        if len(self.gaze_history) == 0:
            return h_ratio, v_ratio

        # Weighted average (recent frames have more weight)
        weights = np.linspace(0.1, 1.0, len(self.gaze_history))
        sum_weights = np.sum(weights)

        avg_h = sum(h * w for (h, v), w in zip(self.gaze_history, weights)) / sum_weights
        avg_v = sum(v * w for (h, v), w in zip(self.gaze_history, weights)) / sum_weights

        return avg_h, avg_v

    def run(self):
        cv2.namedWindow("Gaze Tracking", cv2.WINDOW_NORMAL)

        while True:
            _, frame = self.webcam.read()
            self.gaze.refresh(frame)
            frame = self.gaze.annotated_frame()

            if self.calibrating:
                frame = self.perform_calibration(frame)

                key = cv2.waitKey(1)
                if key == 32:  # SPACE to capture point
                    if self.gaze.pupils_located:
                        h_ratio = self.gaze.horizontal_ratio()
                        v_ratio = self.gaze.vertical_ratio()

                        if h_ratio is not None and v_ratio is not None:
                            self.calibration_data.append({
                                'screen': self.calibration_points[self.current_calibration_point],
                                'gaze': (h_ratio, v_ratio)
                            })

                            self.current_calibration_point += 1
                            if self.current_calibration_point >= len(self.calibration_points):
                                self.calculate_mapping()
                                self.calibrating = False
            else:
                if self.gaze.pupils_located:
                    h_ratio = self.gaze.horizontal_ratio()
                    v_ratio = self.gaze.vertical_ratio()

                    if h_ratio is not None and v_ratio is not None:
                        # Apply smoothing
                        h_ratio, v_ratio = self.smooth_gaze(h_ratio, v_ratio)

                        # Map to screen coordinates
                        screen_x, screen_y = self.map_to_screen(h_ratio, v_ratio)

                        # Move the mouse pointer
                        pyautogui.moveTo(screen_x, screen_y)

                        # Draw gaze point
                        cv2.circle(frame, (screen_x, screen_y), 15, (0, 0, 255), -1)

                        # Check if the user is in the pause/play zone
                        if (self.pause_play_zone["h_range"][0] <= h_ratio <= self.pause_play_zone["h_range"][1] and
                                self.pause_play_zone["v_range"][0] <= v_ratio <= self.pause_play_zone["v_range"][1]):
                            # Detect double blink
                            if self.gaze.is_blinking():
                                current_time = time.time()
                                if current_time - self.last_blink_time < 1:  # Reduced time window to 0.5 seconds
                                    self.blink_count += 1
                                else:
                                    self.blink_count = 1  # Reset blink count if too much time has passed

                                self.last_blink_time = current_time

                                if self.blink_count == 2:  # Double blink detected
                                    self.toggle_pause_play()
                                    self.blink_count = 0  # Reset blink count after action
                        else:
                            self.blink_count = 0  # Reset blink count if not in the zone

                        # Display the pause/play zone on the screen
                        zone_x1 = int(self.pause_play_zone["h_range"][0] * frame.shape[1])
                        zone_x2 = int(self.pause_play_zone["h_range"][1] * frame.shape[1])
                        zone_y1 = int(self.pause_play_zone["v_range"][0] * frame.shape[0])
                        zone_y2 = int(self.pause_play_zone["v_range"][1] * frame.shape[0])
                        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 0), 2)
                        cv2.putText(frame, "Pause/Play Zone", (zone_x1, zone_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Gaze Tracking", frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC to exit
                break
            elif key == ord('r'):  # Reset calibration
                self.calibrating = True
                self.current_calibration_point = 0
                self.calibration_data = []
                self.calibration_complete = False
                self.gaze_history.clear()

        self.webcam.release()
        cv2.destroyAllWindows()

    def toggle_pause_play(self):
        """Toggle between pause and play states."""
        if self.is_playing:
            pyautogui.press('space')  # Simulate spacebar press to pause
            self.is_playing = False
            print("Paused")
        else:
            pyautogui.press('space')  # Simulate spacebar press to play
            self.is_playing = True
            print("Playing")

tracker = EnhancedGazeTracker()
tracker.run()