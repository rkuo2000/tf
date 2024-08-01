## pip install ultralytics
# usage: python yolov10.py images/traffic.jpg

from ultralytics import YOLO
import sys

if len(sys.argv) >1:
    filename = sys.argv[1]
else:
    filename = "images/traffic.jpg"

model = YOLO("yolov10n.pt")

results = model(filename) # ,save-txt=True)

results[0].show()

labels = results[0].names

cls = results[0].boxes.cls.tolist()

unique = list(dict.fromkeys(cls))

text = "There are "
for label in unique:
    count = cls.count(label)
    text = text + str(count) + " " + labels[int(label)] + ","
print(text)
