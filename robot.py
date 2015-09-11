# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
import re

w = 640
h = 480

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)*30

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('img-1/*.jpg') # img-1 o img-2
# images = glob.glob('calib1Sep/Debug/Images2/*.jpg')
# print images


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,7), corners,ret)
        # cv2.imshow('img',img)
        name = re.findall(r'[\w\d\-]+\.\w+',fname)[0]
        cv2.imwrite('reporte/'+name,img)
        # cv2.waitKey(500)

# cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

### Undistortion

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

for fname in images:
    img = cv2.imread(fname)
    name = re.findall(r'[\w\d\-]+\.\w+',fname)[0]
    # h, w = img.shape[:2]
    # print h, w
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    # print roi
    # dst = dst[y:y+h, x:x+w]
    cv2.imwrite('reporte/Distorsión/'+name,dst)
'''
mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "total error: ", mean_error/len(objpoints)
'''
