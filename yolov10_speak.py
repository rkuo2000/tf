## pip install ultralytics
## pip install gtts
## pip install mpg123
# usage: python yolov10.py images/traffic.jpg

import sys
from ultralytics import YOLO
from gtts import gTTS
from mpg123 import Mpg123, Out123

if len(sys.argv) >1:
    filename = sys.argv[1]
else:
    filename = "images/traffic.jpg"

## Object Detection
model = YOLO("yolov10n.pt")

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
mp3 = Mpg123("gTTS.mp3")
out = Out123()
for frame in mp3.iter_frames(out.start):
    out.play(frame)
