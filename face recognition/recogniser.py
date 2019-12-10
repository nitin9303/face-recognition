import cv2
import numpy as np

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/trainingdata.yml')
id = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


while(True):
    #img = cv2.imread("D:\\till 7 feb\\FB_IMG_1506742765576.jpg")

    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf = rec.predict(gray[y:+y+h,x:x+w])
        if(id==1):
            id = 'nitin'
        if(id==3):
            id = 'arun'
        elif(id==2):
            id = 'satender'
        cv2.putText(img,str(id),(x,y+h),font,1,(255,0,0))
        
    cv2.imshow('face',img)
    if cv2.waitKey(1) ==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
