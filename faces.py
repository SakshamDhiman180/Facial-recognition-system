import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer= cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name":1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
              
cap=cv2.VideoCapture(0)
cap.set(3,1280) # set Width
cap.set(4,720) # set Height

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    rval, frame = cap.read()
    frame = cv2.flip(frame,0)
    frame = cv2.flip(frame,-1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5 ,minNeighbors=5)
    # Display the resulting frame
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_,conf = recognizer.predict(roi_gray)
        if conf>=45:# and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            


        #img_item="7.png"
        #cv2.imwrite(img_item,roi_color)

        color = (0,255,0)
        stroke = 1
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
        
    cv2.imshow('frame',frame)

    K = cv2.waitKey(30) & 0xFF
    if K==27:
        break
cap.release()
cv2.destroyAllWindows()
