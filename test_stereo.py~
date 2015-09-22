import numpy as np
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)


M = 64
m = 32
n = M - m
w = 5

with np.load('data/maps.npz') as X:
    map1L, map2L, map1R, map2R = [X[i] for i in ('arr_0', 'arr_1', 'arr_2', 'arr_3')]

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    imgL = cv2.remap(frame, map1L, map2L, cv2.INTER_LINEAR)
    imgR = cv2.remap(frame2, map1R, map2R, cv2.INTER_LINEAR)

    stereo = cv2.StereoSGBM(m, n, w, 8*w*w, 32*w*w, 1, 54, 14, 160, 1)
    imgL = cv2.pyrDown(imgL)
    imgR = cv2.pyrDown(imgR)

    disparity = stereo.compute(imgL,imgR)
    dmap = disparity.astype('uint8')
    cv2.imshow('Depth map', cv2.applyColorMap(-dmap, cv2.COLORMAP_JET))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
