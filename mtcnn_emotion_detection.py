### fer2013_cnn.h5 downloaded from https://kaggle.com/rkuo2000/fer2013-cnn/data to ~/tf/models
### make sure Kaggle Tensorflow version is the same as this local machine to run
# $pip install tensorflow --upgrade
# $pip install mtcnn
# $python mtcnn_emotion_detection image/facemask1.jpg
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mtcnn import MTCNN
from tensorflow.keras import models

# read image
if len(sys.argv)>1:
    img = plt.imread(sys.argv[1])
else:
    img = plt.imread("images/facemask1.jpg")

#
# Face Detection : MTCNN
#
detector = MTCNN()

faces = detector.detect_faces(img)

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

# draw box for detected face on image
for box in bbox:
    x,y,w,h = box

    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) # draw box, thickness=2

#
# Mask Detection : FaceMask-CNN
#

# load FaceMask-CNN model
model = models.load_model('models/fer2013_cnn.h5')

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

for box in bbox:
    # get coordinate
    print(box)
    x,y,w,h = box
    if x<0: x=0
    if y<0: y=0	
    # get region of interest
    roi = img[y:y+h, x:x+w]             # get face image
    print(len(roi))
    print(type(roi))
    roi = cv2.resize(roi,(48,48))       # opencv resize to 96x96
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) # convert color to gray
	 
    # normalized data & reshape it
    x_test = roi/255.0                  # convert to floating point
    x_test = x_test.reshape(-1,48,48,1) # reshape for model input
    
    # mask detection
    preds = model.predict(x_test)
    maxindex = int(np.argmax(preds))
    txt = labels[maxindex]
    acc = str(int(preds[0][maxindex] * 100))+'%'
    
    # draw box for detected face on image
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) # draw box, thickness=2
    cv2.putText(img, txt, (x,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) # font scale=0.5, thickness=2
    cv2.putText(img, acc, (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) # font scale=0.5, thickness=2

# display detected image
plt.axis('off')
plt.imshow(img)
plt.show()
   
# save image
plt.imsave('out/detected.jpg', img)
