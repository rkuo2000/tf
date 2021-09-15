# use camera to classify red bean
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import models

camera = cv2.VideoCapture(0)

# check camera resolution
(_, frame) = camera.read()
(height, width, channel) = frame.shape
print(width, height)

# Load Model
model = models.load_model('models/redbean_cnn.h5')

# Dictionary
dict = {0: 'crack', 1: 'good'}

while True:
    (_, frame) = camera.read()
    frame = cv2.flip(frame, 0) # vertical flip	
    frame = cv2.flip(frame, 1) # horizontal flip
	
    top, bottom, left, right = 20, 148, 200, 328 # 128x128
    roi = frame[top:bottom, left:right]          # region of interest
        	
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
	
    x_data = roi / 255.0
    x_data = x_data.reshape(1,128,128,3)

    # prediction
    predictions = model.predict(x_data)
    maxindex = int(np.argmax(predictions))
    print(predictions[0][maxindex], dict[maxindex])
	
    cv2.putText(frame, dict[maxindex], (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
    cv2.putText(frame, str(predictions[0][maxindex]), (left, bottom+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)       
    cv2.imshow('frame', frame)
	
    keypress = cv2.waitKey(1) & 0xFF # keypress by user 
    if keypress == ord("q"): # press q to quit
        break
		
# free up memory
camera.release()
cv2.destroyAllWindows()
