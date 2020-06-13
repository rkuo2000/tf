### Emotion Detection 
# Usage : python3 emotion_detection_resnet.py --mode train
#         python3 emotion_detection_resnet.py --mode detect
import numpy as np
import argparse
import matplotlib.pyplot as plt
import datetime
import os
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/detect")
a = ap.parse_args()
mode = a.mode 

# Define data generators
train_dir = 'dataset/fer2013-clean/train'
val_dir = 'dataset/fer2013-clean/val'

num_classes = 7
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Build Model
net = ResNet50(weights=None, include_top=False, input_shape=(48,48,1))
x = net.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.25)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=net.input, outputs=output_layer)
model.summary()

# To train model, use argument "--mode train"
if mode == "train":
    print(datetime.datetime.now())	
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
	
    if os.path.isfile('model/fer2013_resnet.h5'):
            model.load_weights('model/fer2013_resnet.h5')
			
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=val_generator,
            validation_steps=num_val // batch_size)
    print(datetime.datetime.now())
    model.save_weights('model/fer2013_resnet.h5')

# To detect by webcam, use argument "--mode detect"
elif mode == "detect":
    model.load_weights('model/emotion_detection_resnet.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_labels = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_labels[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
