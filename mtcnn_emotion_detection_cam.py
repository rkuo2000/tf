### fer2013_cnn.h5 downloaded from https://kaggle.com/rkuo2000/fer2013-cnn/data to ~/tf/models
### make sure Kaggle Tensorflow version is the same as this local machine to run
# $pip install mtcnn
import cv2
import sys
import numpy as np

from mtcnn import MTCNN
from tensorflow.keras import models

# load MTCNN model
detector = MTCNN()
	
# load FaceMask-CNN model
model = models.load_model('models/fer2013_cnn.h5')

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

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
        if confidence>=0.9: # check confidence > 90%
            bbox.append(box)

    for box in bbox:
        # get coordinate
        x,y,w,h = box
        if x<0: x=0
        if y<0: y=0
        # get region of interest
        roi = frame[y:y+h, x:x+w]           # get face image
        roi = cv2.resize(roi,(48,48))       # opencv resize to 48x48
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # normalized data & reshape it
        x_test = roi/255.0                  # convert to floating point
        x_test = x_test.reshape(-1,48,48,1) # reshape for model input

        # mask detection
        preds = model.predict(x_test)
        maxindex = int(np.argmax(preds))
        txt = labels[maxindex]
        acc = str(int(preds[0][maxindex] * 100))+'%'

        # draw box for detected face on image
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2) # draw box, thickness=2
        cv2.putText(frame, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) # font scale=1, thickness=2
        cv2.putText(frame, acc, (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) # font scale=1, thickness=2

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
