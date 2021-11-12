### facemask_cnn.h5 downloaded from https://kaggle.com/rkuo2000/facemask-cnn/data to ~/tf/models
### make sure Kaggle Tensorflow version is the same as this local machine to run
# $pip install mtcnn
# $python mtcnn_facemask_cam.py
import cv2
import sys
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras import models

# load MTCNN model
detector = MTCNN()

# load FaceMask-CNN model
model = models.load_model('models/facemask_cnn.h5')
labels = ['With Mask', 'Without Mask']

# get video
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    faces = detector.detect_faces(frame)

    # find bounding box of each face (with confidence>90%)
    bbox = []
    for face in faces:
        box = face['box']
        print(box)
        keypoints= face['keypoints']
        print(keypoints)
        confidence= face['confidence']
        print(confidence)
        print()
        if confidence>=0.9:
            bbox.append(box)

    for box in bbox:
        # get coordinate
        x,y,w,h = box
        # get region of interest
        roi = frame[y:y+h, x:x+w]           # get face image
        roi = cv2.resize(roi,(96,96))       # opencv resize to 96x96
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # normalized data & reshape it
        x_test = roi/255.0                  # convert to floating point
        x_test = x_test.reshape(-1,96,96,3) # reshape for model input

        # mask detection
        preds = model.predict(x_test)
        maxindex = int(np.argmax(preds))
        txt = labels[maxindex]
        acc = str(int(preds[0][maxindex] * 100))+'%'

        # draw box for detected face on image
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2) # draw box, thickness=2
        cv2.putText(frame, txt, (x,y+10), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2) # font scale=1, thickness=2
        cv2.putText(frame, acc, (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) # font scale=1, thickness=2

    cv2.imshow('FaceMask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
