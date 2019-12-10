import cv2
import os
import numpy as np

def facedetection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=2,minNeighbors=7)

    return faces,gray_img

test_img = cv2.imread("D:\\till 7 feb\\FB_IMG_1506742765576.jpg")
face_detected,gray_img = facedetection(test_img)
print('face_detected:',face_detected)

for (x,y,w,h) in face_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=1)

resized_img = cv2.resize(test_img,(1000,700))
cv2.imshow('face dection',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

