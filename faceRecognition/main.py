import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib 
from keras_facenet import FaceNet


#INITIALIZE
facenet = FaceNet()
faces_embeddings,Y_labels = joblib.load("face_embeddings_with_labels.joblib")
encoder = LabelEncoder()
print(Y_labels)
encoder.fit(Y_labels)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = joblib.load('faceRecognitionModel.joblib')

# print("YES")
cap = cv.VideoCapture(0)

while cap.isOpened():
    # The standard method for detecting faces using haarcascade
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)    
    for x,y,w,h in faces:
        # Extracting only the face window
        img = rgb_img[y:y+h, x:x+w]
        # Preprocessing for our model
        img = cv.resize(img, (160,160)) 
        img = np.expand_dims(img,axis=0)
        ypred = facenet.embeddings(img)

        # Get the probabilities of all the classes
        probability = model.predict_proba(ypred)

        # If the probability is higher than 0.65 use the model prediction, else print unknown
        final_name = encoder.inverse_transform([np.argmax(probability)]) if np.max(probability)>0.65 else 'Unknown'
        
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') ==27:
        break

cap.release()
cv.destroyAllWindows