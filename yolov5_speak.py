## pip install ultralytics
## pip install gtts
# usage: python yolov5.py images/traffic.jpg

import os
import sys
from ultralytics import YOLO
from gtts import gTTS

if len(sys.argv) >1:
    filename = sys.argv[1]
else:
    filename = "images/traffic.jpg"

## Object Detection
model = YOLO("yolov5n.pt")

results = model(filename) # ,save-txt=True)

results[0].show()

## Object Counting
labels = results[0].names

cls = results[0].boxes.cls.tolist()

unique = list(dict.fromkeys(cls))

text = "There are "
for label in unique:
    count = cls.count(label)
    text = text + str(count) + " " + labels[int(label)] + ","
print(text)

## TTS 
sl = "en"
tts = gTTS(text, lang=sl)
tts.save("gTTS.mp3")

## Speak
os.system("mpg123 -q gTTS.mp3")
