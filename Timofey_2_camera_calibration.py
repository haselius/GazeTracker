import numpy as np
import cv2
import glob
from utils import draw_grid

# Define pattern size you generated and camera resoution
pattern_size = (7,10)
frameSize = (1920, 1080)

# Termination criteria for calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for pattern, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = []  # 3d points of the pattern
imgpoints = []  # 2d points detected on frame

images = glob.glob(f'./data/calibrations/*.png')

for img in images:

    img_cv = cv2.imread(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret == True:

        objpoints.append(objp)

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners with index of each corner
        ### DO NOT DELETE THIS ###
        ### your code is here ###
        # cv2.drawChessboardCorners()
        # pass
        cv2.drawChessboardCorners(img_cv,pattern_size,corners,ret)
        pass
        cv2.imshow('Img', img_cv)
        # cv2.waitKey(0)
        ### your code is here ###
        ### DO NOT DELETE THIS ###

cv2.destroyAllWindows()

# Calibration results

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
print("K", cameraMatrix)
print("Distortion coefficients", dist)

cv_file = cv2.FileStorage('./data/calib_cam.xml', cv2.FILE_STORAGE_WRITE)
cv_file.write("K", cameraMatrix)
cv_file.write("dist", dist)
cv_file.release()

# Examine distortion effects
h, w = img_cv.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

img = draw_grid(img_cv, color=(0, 255, 0), thickness=1)
cv2.imwrite('./data/utils/orig.png', img)

undst = cv2.undistort(img, cameraMatrix, dist, None, newcameramtx)
cv2.imwrite('./data/utils/undst.png', undst)


