import numpy as np
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt

def nothing(x):
    pass

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

M = 64

with np.load('data/maps.npz') as X:
    map1L, map2L, map1R, map2R = [X[i] for i in ('arr_0', 'arr_1', 'arr_2', 'arr_3')]

cv2.namedWindow('Options', cv2.WINDOW_NORMAL)
cv2.namedWindow('Depth Map')
cv2.namedWindow('Image')
cv2.createTrackbar('minDisparity', 'Options', 1, M-1, nothing)
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

args = [None, None, None, None, None, None, None, None, None, 0]

while True:
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    imgL = cv2.remap(frame, map1L, map2L, cv2.INTER_LINEAR)
    imgR = cv2.remap(frame2, map1R, map2R, cv2.INTER_LINEAR)
    imgL = cv2.pyrDown(imgL)
    imgR = cv2.pyrDown(imgR)
    
    if args[9] == 0:
        args = [cv2.getTrackbarPos('minDisparity','Options'),
                cv2.getTrackbarPos('SADWindowSize','Options'),
                cv2.getTrackbarPos('P1','Options'),
                cv2.getTrackbarPos('P2','Options'),
                cv2.getTrackbarPos('disp12MaxDiff','Options'), 
                cv2.getTrackbarPos('preFilterCap','Options'),
                cv2.getTrackbarPos('uniquenessRatio','Options'),
                cv2.getTrackbarPos('speckleWindowSize','Options'),
                cv2.getTrackbarPos('speckleRange','Options'),
                cv2.getTrackbarPos(switch,'Options')
               ]
        stereo = cv2.StereoSGBM(16*args[0],
                                16* (M - args[0]),
                                2*args[1] + 1,
                                args[2],
                                args[3],
                                args[4],
                                args[5],
                                args[6],
                                args[7],
                                args[8])
        
    else:
        disparity = stereo.compute(imgL,imgR)
        dmap = disparity.astype('uint8')
        cv2.imshow('Depth Map', cv2.applyColorMap(-dmap, cv2.COLORMAP_JET))
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
