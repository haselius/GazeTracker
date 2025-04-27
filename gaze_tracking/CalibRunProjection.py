import cv2
import numpy as np
from gaze_tracking.gaze_traking2 import GazeTrackingMediaPipe
from collections import deque
import matplotlib.pyplot as plt
import pyautogui
import time


class EnhancedGazeTracker:
    def __init__(self):
        self.gaze = GazeTrackingMediaPipe()

        self.webcam_width, self.webcam_height = 1280, 720 # Webcam parameters

        self.webcam = cv2.VideoCapture(0)

        self.webcam.set(cv2.CAP_PROP_FPS, 30)  # Set webcam frame rate to 30 FPS
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_width)  # Webcam resolution width
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_height)  # Webcam resolution height

        self.screen_width, self.screen_height = 2560, 1440  # Adjust to your display

        # Calibration settings: normalized screen positions
        self.calibration_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        self.calibration_data = []  # Will store pairs: {'gaze': (h_ratio, v_ratio), 'screen': (norm_x, norm_y)}
        self.current_calibration_point = 0
        self.calibrating = True

        # Smoothing
        self.gaze_history = deque(maxlen=5)
        self.calibration_complete = False

        # The computed homography matrix (from webcam gaze coordinates to screen coordinates)
        self.homography_matrix = None

        # Animation parameters (dot moving in circle)
        self.anim_start_time = None
        self.anim_duration = 10

        # Data recording for metric calculation
        self.timestamps = []
        self.dot_positions = []
        self.gaze_positions = []
        self.star_time = None

    def perform_calibration(self, frame):
        point = self.calibration_points[self.current_calibration_point]
        x = int(point[0] * frame.shape[1])
        y = int(point[1] * frame.shape[0])
        # Draw the calibration point on the frame
        cv2.circle(frame, (x, y), 15, (0, 255, 0), 2)
        cv2.putText(frame, f"Look at the circle ({self.current_calibration_point + 1}/{len(self.calibration_points)})",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE when ready", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def calculate_mapping(self):
        """
        Compute the homography (perspective transform) from the gaze space to the screen.
        - Source points: the collected gaze coordinates (scaled to webcam resolution)
        - Destination points: the known screen positions (scaled from normalized calibration points)
        """
        src_points = []
        dst_points = []

        for data in self.calibration_data:
            # Gaze coordinates in the webcam coordinate system
            gaze_x = data['gaze'][0] *  self.webcam_width
            gaze_y = data['gaze'][1] *  self.webcam_height
            src_points.append([gaze_x, gaze_y])
            # Screen coordinates from calibration_points, scaled to your screen dimensions
            screen_x = data['screen'][0] * self.screen_width
            screen_y = data['screen'][1] * self.screen_height
            dst_points.append([screen_x, screen_y])

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)

        # print("Calibration source points (webcam):", src)
        # print("Calibration destination points (screen):", dst)

        self.homography_matrix, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        # print("Computed Homography Matrix:\n", self.homography_matrix)
        # print("Homography status array:", status)
        if self.homography_matrix is not None:
            self.calibration_complete = True
        else:
            print("Error: Homography matrix not computed.")

    def map_to_screen(self, h_ratio, v_ratio):
        """
        Map the current gaze (normalized ratios) to screen coordinates using the computed homography.
        If calibration is not complete, it falls back to a simple proportional mapping.
        """
        if not self.calibration_complete or self.homography_matrix is None:
            return int(h_ratio * self.screen_width), int(v_ratio * self.screen_height)

        # Create source point in webcam coordinate system
        src_point = np.array([[[h_ratio * self.webcam_width, v_ratio * self.webcam_height]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, self.homography_matrix)
        x = int(dst_point[0][0][0])
        y = int(dst_point[0][0][1])

        # Clamp coordinates to the screen boundaries
        x = max(0, min(self.screen_width, x))
        y = max(0, min(self.screen_height, y))


        # print(f"Mapping gaze ({h_ratio:.3f}, {v_ratio:.3f}) -> Screen ({x}, {y})")
        return x, y

    def smooth_gaze(self, h_ratio, v_ratio):
        """Apply smoothing to reduce overdrift due to calibration. At least try("""
        self.gaze_history.append((h_ratio, v_ratio))
        if not self.gaze_history:
            return h_ratio, v_ratio

        weights = np.linspace(0.1, 1.0, len(self.gaze_history))
        sum_weights = np.sum(weights)
        avg_h = sum(h * w for (h, _), w in zip(self.gaze_history, weights)) / sum_weights
        avg_v = sum(v * w for (_, v), w in zip(self.gaze_history, weights)) / sum_weights

        return avg_h, avg_v

    def metric(self, frame):

        if self.anim_start_time is None:
            self.anim_start_time = time.time()


        dot_color = (0, 0, 255)
        dot_radius = 10

        time_start = (time.time() - self.anim_start_time) % self.anim_duration
        theta = 2 * np.pi * (time_start / self.anim_duration)

        # Compute dot position
        height, width = frame.shape[:2]
        cx, cy = width // 2, height // 2
        radius = min(width, height) // 2.5
        x = int(cx + radius * np.cos(theta))
        y = int(cy + radius * np.sin(theta))

        # Draw the moving dot
        cv2.putText(frame, "Track the red circle", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), dot_radius, dot_color, -1)
        return x, y

    def plot_and_save(self, filename='delta_plot.png'):
        # Compute deltas and plot
        times = self.timestamps
        dots = np.array(self.dot_positions)
        gazes = np.array(self.gaze_positions)
        deltas = np.linalg.norm(dots - gazes, axis=1)

        plt.figure()
        plt.plot(times, deltas)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (pixels)')
        plt.title('Gaze vs. Dot Position Error')
        plt.savefig(filename)
        print(f"Delta plot saved to {filename}")



    def run(self):
        cv2.namedWindow("Gaze Tracking", cv2.WINDOW_NORMAL)
        pyautogui.FAILSAFE = False



        while True:
            ret, frame = self.webcam.read()
            if not ret:
                break

            self.gaze.refresh(frame)
            frame = self.gaze.annotated_frame()


            if self.calibrating:
                frame = self.perform_calibration(frame)
                key = cv2.waitKey(1)
                if key == 32:  # SPACE to capture calibration point
                    if self.gaze.pupils_located:
                        h_ratio = self.gaze.horizontal_ratio()
                        v_ratio = self.gaze.vertical_ratio()
                        # print( f"Captured calibration point {self.current_calibration_point}: gaze=({h_ratio:.3f}, {v_ratio:.3f})")
                        if h_ratio is not None and v_ratio is not None:
                            self.calibration_data.append({
                                'gaze': (h_ratio, v_ratio),
                                'screen': self.calibration_points[self.current_calibration_point]
                            })
                            self.current_calibration_point += 1
                            if self.current_calibration_point >= len(self.calibration_points):
                                self.calculate_mapping()
                                self.calibrating = False
            else:
                if self.star_time is None:
                    self.star_time = time.time()
                x_dot, y_dot = self.metric(frame)
                t = time.time() - self.star_time
                if self.gaze.pupils_located:
                    h_ratio = self.gaze.horizontal_ratio()
                    v_ratio = self.gaze.vertical_ratio()
                    if h_ratio is not None and v_ratio is not None:

                        # Smooth the gaze data
                        h_ratio, v_ratio = self.smooth_gaze(h_ratio, v_ratio)
                        # Map gaze to screen coordinates using the homography
                        screen_x, screen_y = self.map_to_screen(h_ratio, v_ratio)
                        # Move the mouse pointer
                        pyautogui.moveTo(screen_x, screen_y)
                        # Draw the gaze point on the frame for visualization
                        cv2.circle(frame, (screen_x, screen_y), 15, (0, 0, 255), -1)

                        self.timestamps.append(t)
                        self.dot_positions.append((x_dot, y_dot))
                        self.gaze_positions.append((screen_x, screen_y))

            cv2.imshow("Gaze Tracking", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC to exit
                break
            elif key == ord('r'):  # Reset calibration
                # Reset all
                self.calibrating = True
                self.current_calibration_point = 0
                self.calibration_data.clear()
                self.calibration_complete = False
                self.gaze_history.clear()
                self.anim_start_time = None
                self.timestamps.clear()
                self.dot_positions.clear()
                self.gaze_positions.clear()
        self.plot_and_save()
        self.webcam.release()
        cv2.destroyAllWindows()

    # def toggle_pause_play(self):
    #     """Toggle between pause and play states by simulating a spacebar press."""
    #     if self.is_playing:
    #         pyautogui.press('space')
    #         self.is_playing = False
    #         print("Paused")
    #     else:
    #         pyautogui.press('space')
    #         self.is_playing = True
    #         print("Playing")


tracker = EnhancedGazeTracker()
tracker.run()