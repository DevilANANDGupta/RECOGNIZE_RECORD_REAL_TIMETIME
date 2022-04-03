import tensorflow as tf
from cvzone.FaceMeshModule import FaceMeshDetector
import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
import pyautogui
from keras import Sequential
# model= Sequential()
SCREEN_SIZE = tuple(pyautogui.size())
fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = 12.0
out = cv2.VideoWriter("output.avi", fourcc, fps, (SCREEN_SIZE))
record_seconds = 10

detector = FaceMeshDetector(maxFaces=2)


def predict_classes(self, x, batch_size=32, verbose=1):
        
        
        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.shape[-3] > 1:
            return proba.argmax(axis=-3)
        else:
            return (proba > 0.5).astype('int32') 



facedetect = cv2.CascadeClassifier("C:/Users/HP/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_SIMPLEX
# model = load_model("D:/facial_recognition/images/anand_chutiya.h5")
model = load_model("D:/facial_recognition/images/new_model.h5")
#take class name
classes = ['ANAND', 'ANAND1','ANAND2']
print(classes[0])

def get_className(classNo):
    if classNo == 0:
        return"anand"
    elif classNo==1:
        return"puja"
    elif classNo==2:
        return"priti"
    elif classNo==3:
        return"NONE"
while True:
    sucess, imgOriginal= cap.read()
    imgOriginal, faces = detector.findFaceMesh(imgOriginal)
    if faces:
        print(faces[0])
        
    
    faces = facedetect.detectMultiScale(imgOriginal, 1.3,5)
    for x,y,w,h in faces:
        crop_img=imgOriginal[y:y+h,x:x+w]
        img=cv2.resize(crop_img, (224,224))
        img=img.reshape(1, 224, 224,3)
        prediction = model.predict(img)
        # model=sequential()
        # predictions = model.predict((img.shape[0],img.shape[1],img.shape[2]))

        # classIndex= classes[np.argmax(img)]
        classIndex = model.predict_classes(img)
        # classIndex=model.classIndex1
        # np.argmax(model.predict(x), axis=-1)
        # classIndex = np.argmax(model.predicted_classes(img), axis=-1)
        probabilityValue = np.amax(prediction)
        if classIndex==0:
            cv2.rectangle(imgOriginal, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.rectangle(imgOriginal, (x,y-40), (x+w, y),(0,255,0),-2)
            cv2.putText(imgOriginal,str(get_className(classIndex)),(x,y-20), font,0.75,(255,0,255))
        elif classIndex==1:
            cv2.rectangle(imgOriginal, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.rectangle(imgOriginal, (x,y-40), (x+w, y),(0,255,0),-2)
            cv2.putText(imgOriginal,str(get_className(classIndex)),(x,y-20), font,0.75,(255,0,255))
        elif classIndex==2:
            cv2.rectangle(imgOriginal, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.rectangle(imgOriginal, (x,y-40), (x+w, y),(0,255,0),-2)
            cv2.putText(imgOriginal,str(get_className(classIndex)),(x,y-20), font,0.75,(255,0,255))
        elif classIndex==3:
            cv2.rectangle(imgOriginal, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.rectangle(imgOriginal, (x,y-40), (x+w, y),(0,255,0),-2)
            cv2.putText(imgOriginal,str(get_className(classIndex)),(x,y-20), font,0.75,(255,0,255))
        cv2.putText(imgOriginal,str(round(probabilityValue*100, 2))+'%', (180, 95), font, 0.75, (255,255,0))
        
    cv2.imshow("RESULT",imgOriginal)
    
    k=cv2.waitKey(10)
    
    for i in range(int(record_seconds * fps)):
            # make a screenshot
                img = pyautogui.screenshot()
            # convert these pixels to a proper numpy array to work with OpenCV
                frame = np.array(img)
            # convert colors from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # write the frame
                out.write(frame)
                cv2.imshow("screenshot", frame)
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()