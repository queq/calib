#!/usr/bin/env python

import numpy as np
import cv2

def nothing(x):
	pass

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

max_disp = 48 / 16

with np.load('data/maps.npz') as X:
	map1L, map2L, map1R, map2R = [X[i] for i in ('arr_0', 'arr_1', 'arr_2', 'arr_3')]

cv2.namedWindow('Options', cv2.WINDOW_NORMAL)
cv2.namedWindow('Disparity')
cv2.namedWindow('Images')
cv2.createTrackbar('minDisparity', 'Options', 1, max_disp-1, nothing)
cv2.createTrackbar('SADWindowSize', 'Options', 0, 5, nothing)
cv2.createTrackbar('P1', 'Options', 6, 450, nothing)
cv2.createTrackbar('P2', 'Options', 0, 2000, nothing)
cv2.createTrackbar('disp12MaxDiff', 'Options', 0, 3, nothing)
cv2.createTrackbar('preFilterCap', 'Options', 22, 63, nothing)
cv2.createTrackbar('uniquenessRatio', 'Options', 15, 15, nothing)
cv2.createTrackbar('speckleWindowSize', 'Options', 200, 200, nothing)
cv2.createTrackbar('speckleRange', 'Options', 1, 2, nothing)
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'Options', 0, 1, nothing)

_, imgL = cap.read()
_, imgR = cap2.read()
imgL = cv2.remap(imgL, map1L, map2L, cv2.INTER_CUBIC)
imgR = cv2.remap(imgR, map1R, map2R, cv2.INTER_CUBIC)
imgL = cv2.pyrDown(imgL)
imgR = cv2.pyrDown(imgR)

while True:
	args = [cv2.getTrackbarPos('minDisparity','Options'), cv2.getTrackbarPos('SADWindowSize','Options'), cv2.getTrackbarPos('P1','Options'),  cv2.getTrackbarPos('P2','Options'), cv2.getTrackbarPos('disp12MaxDiff','Options'), cv2.getTrackbarPos('preFilterCap','Options'),  cv2.getTrackbarPos('uniquenessRatio','Options'), cv2.getTrackbarPos('speckleWindowSize','Options'), cv2.getTrackbarPos('speckleRange','Options'), cv2.getTrackbarPos(switch,'Options')]
	min_disp = args[0]*16
	num_disp = 16*(max_disp - args[0])
	stereo = cv2.StereoSGBM(minDisparity = min_disp, numDisparities = num_disp, SADWindowSize = 2*args[1]+1, uniquenessRatio = args[6],  speckleWindowSize = args[7], speckleRange = args[8], disp12MaxDiff = args[4], preFilterCap = args[5], P1 = args[2], P2 = args[3])
	disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
	cv2.imshow('Images', np.concatenate((imgL, imgR), axis=1))
	cv2.imshow('Disparity', (disp-min_disp)/num_disp)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
