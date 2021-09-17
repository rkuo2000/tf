### facemask_cnn.h5 downloaded from https://kaggle.com/rkuo2000/facemask-cnn/data to ~/tf/models
### make sure Kaggle Tensorflow version is the same as this local machine to run
# $pip install tensorflow --upgrade
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
labels = ['With Mask', 'No Mask']

# get video
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame,0) # vertical flip
    frame = cv2.flip(frame,1) # horizonal flip
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(image)

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
        roi = image[y:y+h, x:x+w]             # get face image
        print(len(roi))
        print(type(roi))
        roi = cv2.resize(roi,(96,96))       # opencv resize to 96x96
    
        # normalized data & reshape it
        x_test = roi/255.0                  # convert to floating point
        x_test = x_test.reshape(-1,96,96,3) # reshape for model input
    
        # mask detection
        preds = model.predict(x_test)
        maxindex = int(np.argmax(preds))
        txt = labels[maxindex]
        acc = str(int(preds[0][maxindex] * 100))+'%'
    
        # draw box for detected face on image
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2) # draw box, thickness=2
        cv2.putText(image, txt, (x,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) # font scale=0.5, thickness=2
        cv2.putText(image, acc, (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) # font scale=0.5, thickness=2
		
    cv2.imshow('Face Mask Detection', cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    key=cv2.waitKey(100)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
