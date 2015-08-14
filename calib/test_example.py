import numpy as np
import cv2
import glob
import re

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((13 * 12, 3),np.float32)
objp[:, :2] = np.mgrid[0:13, 0:12].T.reshape(-1, 2) * 30

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('calib/example/Image*.tif')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (13, 12), None)
    print ret
    # If found, add object points, image points (after refining them)
    if ret is True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (13, 12), corners2, ret)
        filename = re.findall(r'[\w\d]+\.[\w]+', fname)[0]
        cv2.imwrite("calib/result/res{}".format(filename), img)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)
print mtx
np.savez('calib/data/Data_example.npz', mtx, dist, rvecs, tvecs)

img = cv2.imread('calib/example/Test.tif')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, mtx, dist, None,newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calib/result/calibresult.tif', dst)

mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print "total error: ", mean_error / len(objpoints)
