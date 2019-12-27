import cv2
import sys
import os
import time
from random import *
import numpy as np
from gtts import gTTS
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

camera = cv2.VideoCapture(0)

# check camera resolution
(_, frame) = camera.read()
(height, width, channel) = frame.shape
print(width, height)

# Load Model
#model = load_model('model/paper-rock-scissors.h5')

# Dictionary
dict = {0: 'paper', 1: 'rock', 2: 'scissors'}	

tts = gTTS('剪刀', lang='zh-tw')
tts.save('scissors.mp3')
tts = gTTS('石頭', lang='zh-tw')
tts.save('rock.mp3')
tts = gTTS('布', lang='zh-tw')
tts.save('paper.mp3')
tts = gTTS('平手', lang='zh-tw')
tts.save('tie.mp3')
tts = gTTS('你贏了', lang='zh-tw')
tts.save('you_win.mp3')
tts = gTTS('我贏了', lang='zh-tw')
tts.save('i_win.mp3')
tts = gTTS('準備猜拳', lang='zh-tw')
tts.save('playagain.mp3')
		
while True:
    (_, frame) = camera.read()
    frame = cv2.flip(frame, 1) # flip the frame so that it is not the mirror view
	
    top, bottom, left, right = 20, 244, 20, 244 # 224x224
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

    top, bottom, left, right = 20, 244, 400, 624
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
	
    cv2.imshow('frame', frame)
	
    # say paper-rock-scissors (剪刀-石頭-布)
#    os.system('cmdmp3 scissors.mp3')
#    os.system('cmdmp3 rock.mp3')
    os.system('cmdmp3 paper.mp3')
		
    # user roi
    roi_user = frame[top:bottom, left:right]       	
    roi_gray = cv2.cvtColor(roi_user, cv2.COLOR_BGR2GRAY)
    roi_bgr  = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    frame[top:bottom, left:right] = roi_bgr
    cv2.imshow('captured', roi_bgr)
		
    # computer roi
    x = randint(0, 2)
    print(x)
    if x==0:
        roi_computer = cv2.imread('image/paper.jpg')		
    if x==1:
        roi_computer = cv2.imread('image/rock.jpg')	
    if x==2:
        roi_computer = cv2.imread('image/scissors.jpg')
		
    # computer roi
    top, bottom, left, right = 20, 244, 20, 244		
    frame[top:bottom, left:right] = roi_computer
    cv2.imshow('frame', frame)
	
    # detect user roi
    res = cv2.resize(roi_bgr, (96, 96), interpolation = cv2.INTER_CUBIC)
    x_data = res / 255.0
    x_data = x_data.reshape(1,96,96,3)

    # prediction
    #predictions = model.predict(x_data)
    #maxindex = int(np.argmax(predictions))
    maxindex = 0
    y = maxindex
    #print(predictions[0][maxindex], dict[maxindex])
    print(y)

	# put text on user box
    cv2.putText(frame, dict[maxindex], (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
    #cv2.putText(frame, str(predictions[0][maxindex]), (left, bottom+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)       

    # decide who win the game
    if x==0: #電腦出布
        if y==0: #你出布
            winner='平手'
        if y==1: #你出石頭				
            winner='我贏了'
        if y==2: #你出剪刀
            winner='你贏了'
    if x==1: #電腦石頭
        if y==0: #你出布
            winner='你贏了'
        if y==1: #你出石頭				
            winer='平手'
        if y==2: #你出剪刀
            winner='我贏了'
    if x==2: #電腦剪刀
        if y==0: #你出布
            winner='我贏了'
        if y==1: #你出石頭				
            winner='你贏了'
        if y==2: #你出剪刀
            winner='平手'
	
    # announce winner	
    if winner=='平手':
        os.system('cmdmp3 tie.mp3')
    if winner=='我贏了':
        os.system('cmdmp3 i_win.mp3')
    if winner=='你贏了':
        os.system('cmdmp3 you_win.mp3')
		
	# announce play again		
    os.system('cmdmp3 playagain.mp3')
		
    keypress = cv2.waitKey(1) & 0xFF # keypress by user		
    if keypress == ord("q"): # press q to quit
        break        
		
# free up memory
camera.release()
cv2.destroyAllWindows()
