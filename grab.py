import cv2
import cv2.cv as cv
import numpy as np

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    _, frame2 = cap2.read()
    cv2.imshow('Images', np.concatenate((frame, frame2), axis=1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break
    	
cap.release()
cv2.destroyAllWindows()
