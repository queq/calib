import cv2
import numpy as np
import glob


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img

# Load previously saved data
with np.load('calib/data/Data.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('arr_0', 'arr_1', 'arr_2', 'arr_3')]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * 20

corners3 = np.zeros((7 * 7, 2), np.float32)
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3) * 20
axis2 = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [
                   0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]]) * 20

for fname in glob.glob('calib/img/Snap*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret is True:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        retval, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

        # project 3D points to image plane
        # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis2, rvecs, tvecs, mtx, dist)

        img = drawCube(img, corners2, imgpts)
        cv2.imshow('img', img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite('axis' + fname, img)

cv2.destroyAllWindows()
