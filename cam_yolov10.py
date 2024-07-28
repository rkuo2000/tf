from ultralytics import YOLO
import cv2

model = YOLO("yolov10n.pt")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = model(frame)
    results[0].save("out.jpg")
    img = cv2.imread("out.jpg")
    cv2.imshow('result', img)
    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
