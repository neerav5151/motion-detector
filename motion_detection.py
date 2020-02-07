import cv2
import tkinter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)#enter video location,0 for webcam

ret, frame1 = cap.read()
ret, frame2 = cap.read()
i2 = cv2.absdiff(frame1, frame2)
grayv = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

while cap.isOpened():

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)


    dilate = cv2.dilate(thresh, np.ones((4, 3), dtype=int), iterations=3)
    erode = cv2.erode(dilate, np.ones((4, 3), dtype=int), iterations=3)
    contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 1000:
            continue
        else:
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.drawContours(frame1,contours,-1,(0,255,0),2)

    cv2.imshow('frame', frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
















import cv2
import tkinter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

img=cv2.imread('pic4.png')
imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,thresh=cv2.threshold(imgray,240,255,cv2.THRESH_BINARY)
count,hier=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

for count1 in count:
    approx=cv2.approxPolyDP(count1,0.01*cv2.arcLength(count1,True),True)
    cv2.drawContours(img,[approx],0,(0,0,0),5)
    x=approx.ravel()[0]
    y=approx.ravel()[1]
    print(approx)




cv2.imshow("ok",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
