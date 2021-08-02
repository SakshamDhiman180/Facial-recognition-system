import os
from PIL import Image
import numpy as np
import cv2
import pickle
import time


BASE_DIR=os.path.dirname(os.path.abspath(__file__))# returning the dir path
image_dir = os.path.join(BASE_DIR,"images")# looking for images folder

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer= cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_ids={}
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(image_dir): #iterating through the images file 
    for file in files:
        path= os.path.join(root,file)#getting path of the img
        label = os.path.basename(root).replace(" ","-")#getting label of the img
        #print(label,path)
        if not label in label_ids:
         label_ids[label] = current_id # asigning the no. id to labels 
         current_id += 1

        id_= label_ids[label]
        #print(label_ids)
        #y_labels.append(label)
        #x_train.append(path)
        pil_image=Image.open(path).convert("L")  #converting img into grayscale 
        image_array = np.array(pil_image,"uint8")#converting pixel values in numpy array (image to no.)
        #print(image_array)
        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2 ,minNeighbors=5)

        for (x,y,w,h) in faces:
            roi = image_array[y:y+h,x:x+w]
            x_train.append(roi)
            y_labels.append(id_)

#print(y_labels)
#print(x_train)            

with open("labels.pickle",'wb') as f: #writeing labels in bytes format and saving it in labels.pickle
    pickle.dump(label_ids, f)
    
recognizer.train(x_train, np.array(y_labels))# training cmd
recognizer.save("trainner.yml")#saving the trained data in trainner.yml file
    
print("training done")   
print("***ALL SET TO RECOGNIZE SOME FaCES***")
time.sleep(3.0)
print("open * faces.py * file")
time.sleep(2.0)
