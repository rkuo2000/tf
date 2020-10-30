import cv2
import os
import glob
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

model = keras.models.load_model("model/facemask_cnn.h5")
labels = ['Pass', 'No Mask']

PADDING = 20

def webcam_face_recognizer():
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame

        # We do not want to detect a new identity while the program is in the process of identifying another person
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade)   
        
        key = cv2.waitKey(100)
        cv2.imshow("preview", img)
        cv2.imshow("roi", roi)		

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

def process_frame(img, frame, face_cascade):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global ready_to_detect_identity
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
		
        roi = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (224, 224))
        test_data=roi_resized/255.0
        test_data=test_data.reshape(1,224,224,3)
        prediction = model.predict(test_data)
        maxindex = int(np.argmax(prediction))
        text=labels[maxindex]
        img = cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 255), 2)

    return img


if __name__ == "__main__":
    webcam_face_recognizer()
