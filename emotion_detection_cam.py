### Emotion Detection 
# download fer2013_cnn.h5 from https://kaggle.com/rkuo2000/fer2013-cnn
# put fer2013_cnn.h5 to ~/tf/models
import numpy as np
import cv2
from tensorflow.keras import models

model=models.load_model('models/fer2013_cnn.h5')

labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        preds = model.predict(cropped_img)
        emo = int(np.argmax(preds))
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        cv2.putText(frame, labels[emo], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

