import cv2
import os
import time
import numpy as np
import screeninfo
import datetime
import pandas as pd
from collections import deque
import random
from imutils.video import FPS
import csv
from CalibAndEstimation import EnhancedGazeTracker
from KalmanHelper import KalmanFilterWrapper
def analyze_error_in_pogs(dir, id, pred_x, pred_y, lab_x, lab_y, average_intensity):

    with open(os.path.join(dir, "pogs_analyze_sample.csv"), "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [id, pred_x, pred_y, lab_x, lab_y, average_intensity])

def getScreenSize():
    screen = screeninfo.get_monitors()
    for s in screen:
        if s.is_primary:
            width = s.width
            height = s.height
            width_mm = s.width_mm
            height_mm = s.height_mm
    print(f"Screen Size: {width}x{height}")
    return width, height, width_mm, height_mm

def main():

    dir = "."
    width, height, width_mm, height_mm = getScreenSize()

    time_name = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
    tracker = EnhancedGazeTracker()
    tracker.Calibrun()
    # Initiallize Klalman Filter for better smoothing
    kalman_filter = KalmanFilterWrapper()
    cv2.namedWindow("over", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "over", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    white_frame = 255*np.ones((height, width, 3), dtype=np.uint8)
    
    print(width, height)
    
    num_of_points = 10
    track_x = deque([0] * num_of_points, maxlen=num_of_points)
    track_y = deque([0] * num_of_points, maxlen=num_of_points)

    last_marker_time = time.time()
    recording = False
    points = []

    third_condition_triggered = False
    second_condition_triggered = False

    id = 0
    area_count = 0
    average_intensity = 0
    num_areas = 30
    screen_width = width
    screen_height = height

    area_width = screen_width // 5
    area_height = screen_height // (num_areas // 5)

    points_in_areas = [[] for _ in range(num_areas)]

    area_indices = list(range(num_areas))
    random.shuffle(area_indices)

    fps_frame = FPS().start()

    cap = cv2.VideoCapture(0)#, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    while cap.isOpened():
        try:
            ret, frame = cap.read()
        except StopIteration:
            break

        ### you method here ###
        # Undistort the image
        if not ret:
            break

        # print(tracker.calibrating)
        # frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        x_hat, y_hat = tracker.run(frame)
        x_filtered, y_filtered = kalman_filter.correct_and_predict(x_hat, y_hat)
        # if x_hat == None:
        #     print("None from the run")
        #     break

        ### you method here ###

        cur_time = time.time()

        main_screen = white_frame
        overlay = main_screen.copy()

        track_x.append(x_filtered)
        track_y.append(y_filtered)

        weights = np.arange(1, num_of_points + 1)
        cv2.circle(overlay, (int(np.average(track_x, weights=weights)), int(
            np.average(track_y, weights=weights))), 12, (255, 255, 255), 3)
        cv2.circle(overlay,  (int(np.average(track_x, weights=weights)), int(
            np.average(track_y, weights=weights))), 10, (188, 105, 47), -1)

        for i in range(num_areas):
            area_x = (i % 5) * area_width
            area_y = (i // 5) * area_height
            cv2.rectangle(overlay, (area_x, area_y), (area_x +
                            area_width, area_y + area_height), (192, 192, 192), 1)

        time_elapsed = cur_time - last_marker_time

        if time_elapsed >= 5:
            last_marker_time = cur_time

            print('idxs', area_indices, area_count)

            if area_count == num_areas:
                print('All areas covered. Exiting loop.')
                break

            area_index = area_indices[area_count]

            area_x = (area_index % 5) * area_width
            area_y = (area_index // 5) * area_height

            x = int(area_x + area_width // 2)
            y = int(area_y + area_height // 2)

            points.append((x, y, (0, 0, 255)))
            recording = False
            second_condition_triggered = False
            third_condition_triggered = False
            print('init', recording)

            area_count += 1
            print(f'area_id: {area_index}, {area_count}/{num_areas}')

        elif time_elapsed >= 2 and not recording and points and not second_condition_triggered:
            id += 1
            points[-1] = (points[-1][0], points[-1][1], (0, 204, 0))
            recording = True
            second_condition_triggered = True
            print('green', recording)

        elif time_elapsed >= 4 and not third_condition_triggered and recording and points:
            points[-1] = (points[-1][0], points[-1][1], (0, 140, 255))
            recording = False
            third_condition_triggered = True
            print('red', recording)

        for point in points:
            cv2.drawMarker(
                overlay, (point[0], point[1]), color=point[2], markerType=cv2.MARKER_CROSS, thickness=2)

            if recording:
                analyze_error_in_pogs(dir, id, int(np.average(track_x, weights=weights)), int(
                    np.average(track_y, weights=weights)), point[0], point[1], average_intensity)

        if len(points) > 1:
            points.pop(0)

        fps_frame.update()
        fps_frame.stop()

        cur_fps = fps_frame.fps()

        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(
            overlay,
            f'FPS: {"{:.2f}".format(cur_fps)}',
            (40, 100),
            font,
            1,
            (0, 0, 0),
            1,
        )
        cv2.imshow("over", overlay)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()