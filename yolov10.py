from ultralytics import YOLO

model = YOLO("yolov10n.pt")

results = model("images/cats.jpg")

results[0].show()
