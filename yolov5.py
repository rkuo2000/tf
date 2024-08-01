from ultralytics import YOLO

model = YOLO("yolov5n.pt")

results = model("images/traffic.jpg")

results[0].show()
