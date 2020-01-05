import cv2
import sys
import os
import time
from random import *
import numpy as np
import imutils
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

camera = cv2.VideoCapture(0)

# check camera resolution
(_, frame) = camera.read()
(frame_h, frame_w, frame_ch) = frame.shape
print(frame_w, frame_h)

# Load Model
model = load_model('model/gesture_cnn.h5')

# Dictionary
dict = {0: 'down', 1: 'left', 2: 'right', 3: 'stop', 4: 'up'}

# read images
img = cv2.imread('eye-exam.jpg')
(img_h, img_w, img_ch) = img.shape

imgR = img                      # E to Right
imgU = imutils.rotate(img, 90)  # E to Up
imgL = imutils.rotate(img, 180) # E to Left
imgD = imutils.rotate(img, 270) # E to Down

# computer roi
x = randint(0, 4)
if x==0:
    eyecheck = imgR
    ans = 'right'
if x==1:
    eyecheck = imgU
    ans = 'up'
if x==2:
    eyecheck = imgL
    ans = 'left'
if x==3:
    eyecheck = imgD
    ans = 'down'

	
while True:
    (_, frame) = camera.read()
    frame = cv2.flip(frame, 1) # flip the frame so that it is not the mirror view
	
    top, bottom, left, right = 50, 50+img_h, 50, 50+img_w
    frame[top:bottom, left:right] = eyecheck	
		
    top, bottom, left, right = 20, 244, 308, 532 # 224x224
    roi = frame[top:bottom, left:right]          # region of interest
        	
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
	
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi_rgb  = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
    frame[top:bottom, left:right] = roi_rgb

    res = cv2.resize(roi_rgb, (96, 96), interpolation = cv2.INTER_CUBIC)	
    x_data = res / 255.0    
    x_data = x_data.reshape(1,96,96,3)
    
    # prediction
    predictions = model.predict(x_data)
    maxindex = int(np.argmax(predictions))
    print(predictions[0][maxindex], dict[maxindex])
	
    cv2.putText(frame, dict[maxindex], (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
    cv2.putText(frame, str(predictions[0][maxindex]), (left, bottom+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)       
    if (dict[maxindex]==ans):
        cv2.putText(frame, 'PASS', (int(frame_w/2)-50, frame_h-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2) 
    else:
        cv2.putText(frame, 'FAIL', (int(frame_w/2)-50, frame_h-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2) 	
    cv2.imshow('frame', frame)

    keypress = cv2.waitKey(1) & 0xFF # keypress by user
    if keypress == ord("q"):         # press q to exit
        break
#    if keypress == ord(" "):         # press space to next exam
    if dict[maxindex] == "stop":      # gesture=stop to next exam
        x = randint(0, 4)
        if x==0:
            eyecheck = imgR
            ans = 'right'
        if x==1:
            eyecheck = imgU
            ans = 'up'
        if x==2:
            eyecheck = imgL
            ans = 'left'
        if x==3:
            eyecheck = imgD
            ans = 'down'
	   
# free up memory
cv2.waitKey(0)
camera.release()
cv2.destroyAllWindows()

