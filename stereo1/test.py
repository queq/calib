import numpy as np
import cv2
import cv2.cv as cv
import time as t

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

cap.set(cv.CV_CAP_PROP_FPS, 20)
cap2.set(cv.CV_CAP_PROP_FPS, 20)
i = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    if i % 5 == 0:
        cv2.imwrite('stereo/L{}.png'.format(str(i/5).zfill(4)), frame)
        cv2.imwrite('stereo/R{}.png'.format(str(i/5).zfill(4)), frame2)
    
    i += 1

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Left', gray)
    cv2.imshow('Right', gray2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
