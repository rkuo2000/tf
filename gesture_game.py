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
model = load_model('model/paper-rock-scissors.h5')

# Dictionary
dict = {0: 'paper', 1: 'rock', 2: 'scissors'}	

# generate voice files
tts = gTTS('剪刀,石頭,布', lang='zh-tw')
tts.save('paper-rock-scissors.mp3')
tts = gTTS('平手', lang='zh-tw')
tts.save('tie.mp3')
tts = gTTS('你贏了', lang='zh-tw')
tts.save('you_win.mp3')
tts = gTTS('我贏了', lang='zh-tw')
tts.save('i_win.mp3')
tts = gTTS('再猜一次', lang='zh-tw')
tts.save('playagain.mp3')

# read images
img_paper = cv2.imread('image/paper.jpg')
img_rock  = cv2.imread('image/rock.jpg')
img_scissors = cv2.imread('image/scissors.jpg')

# score
score_computer = 0
score_user = 0

roi_previous = img_paper

while True:
    # computer roi
    x = randint(0, 2)
    if x==0:
        roi_computer = img_paper
    if x==1:
        roi_computer = img_rock
    if x==2:
        roi_computer = img_scissors

    # user roi
    y = 0
    print(x,y)

    while True:
        (_, frame) = camera.read()
        frame = cv2.flip(frame, 1) # flip the frame so that it is not the mirror view
        top, bottom, left, right = 50, 274, 20, 244 # 224x224
        frame[top:bottom, left:right] = roi_previous
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, str(score_computer), (left+100, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
        top, bottom, left, right = 50, 274, 400, 624
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, str(score_user), (left+100, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
        cv2.putText(frame, 'press Space to play, press Q to quit', (20, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),2)
        cv2.imshow('frame', frame)

        keypress = cv2.waitKey(1) & 0xFF # keypress by user
        if keypress == ord(" "):         # press space to play
            break
        if keypress == ord("q"):         # press q to play
            break
    if keypress == ord("q"):
        break
   
    # say paper-rock-scissors (剪刀-石頭-布)
    #os.system('mpg321 paper-rock-scissors.mp3') # for RPi/Linux
    os.system('cmdmp3 paper-rock-scissors.mp3') # for PC
    #os.system('afplay paper-rock-scissors.mp3') # for MAC OS

    # capture frame		
    (_, frame) = camera.read()
    frame = cv2.flip(frame, 1) # flip the frame so that it is not the mirror view

    top, bottom, left, right = 50, 274, 20, 244 # 224x224
    frame[top:bottom, left:right] = roi_computer
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
    roi_previous = roi_computer

    # user roi
    top, bottom, left, right = 50, 274, 400, 624 # 224x224
    roi_user = frame[top:bottom, left:right]       	
    roi_gray = cv2.cvtColor(roi_user, cv2.COLOR_RGB2GRAY)
    roi_rgb  = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

    # detect user roi
    res = cv2.resize(roi_rgb, (96, 96), interpolation = cv2.INTER_CUBIC)
    x_data = res / 255.0
    x_data = x_data.reshape(1,96,96,3)

    # prediction
    predictions = model.predict(x_data)
    maxindex = int(np.argmax(predictions))
    print(predictions[0][maxindex], dict[maxindex])

    # put text on user box
    cv2.putText(frame, dict[maxindex], (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
    cv2.putText(frame, str(predictions[0][maxindex]), (left, bottom+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)       

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
            winner='平手'
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
        #os.system('mpg321 tie.mp3') # for RPi/Linux
        os.system('cmdmp3 tie.mp3') # for PC
        #os.system('afplay tie.mp3') # for MAC OS
    if winner=='我贏了':
        #os.system('mpg321 i_win.mp3') # for RPi/Linux
        os.system('cmdmp3 i_win.mp3') # for PC
        #os.system('afplay i_win.mp3') # for MAC OS
        score_computer+=1
    if winner=='你贏了':
        #os.system('mpg321 you_win.mp3') # for RPi/Linux
        os.system('cmdmp3 you_win.mp3') # for PC
        #os.system('afplay you_win.mp3') # for MAC OS
        score_user+=1

    top, bottom, left, right = 0, 300, 0, 640
    result = frame[top:bottom, left:right]       	
    cv2.imshow('result', result)

    # announce play again		
    #os.system('mpg321 playagain.mp3') # for RPi/Linux
    os.system('cmdmp3 playagain.mp3') # for PC
    #os.system('afplay playagain.mp3') # for MAC OS

# free up memory
camera.release()
cv2.destroyAllWindows()
