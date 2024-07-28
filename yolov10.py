from ultralytics import YOLO

model = YOLO("yolov10s.pt")

results = model("images/traffic.jpg")

results[0].show()
