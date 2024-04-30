import numpy as np
import cv2

cap = cv2.VideoCapture('data/video.mp4')

ret1, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

ret2, frame2 = cap.read()
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(gray1, gray2)

_, thresh = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
thresh_bgr[thresh == 255] = [0, 255, 0]
stacked = np.hstack((frame2, thresh_bgr))

cv2.imwrite('hw0_111550093_2.png', stacked)