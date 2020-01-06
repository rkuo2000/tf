### Emotion Detection using VGG13 
# Usage : python3 emotion_detection_vgg13.py --mode train
#         python3 emotion_detection_vgg13.py --mode detect
import numpy as np
import argparse
import matplotlib.pyplot as plt
import datetime
import os
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/detect")
a = ap.parse_args()
mode = a.mode 

def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()
	
# Define data generators
train_dir = 'data/fer2013/train'
val_dir = 'data/fer2013/val'

num_train = 28709
num_val = 7178
batch_size = 128
num_epoch = 100

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Build Model
model = keras.models.Sequential()
# conv1
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# conv2
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# conv3
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# conv4
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# conv5
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# fc1
model.add(Flatten())
# fc2
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# fc3
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# softmax
model.add(Dense(7, activation='softmax'))
model.summary()
	
def steplr(epoch):
    lr = 0.025
    max_epochs=100.0
    lr = lr * (1.0 - epoch/max_epochs)
    return lr
	
# If you want to train the same model or try other models, go for this
if mode == "train":
    print(datetime.datetime.now())
    sgd = keras.optimizers.SGD(lr=0.025, decay=0.0005, momentum=0.9, nesterov=True)	
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    if os.path.isfile('model/emotion_detection.h5'):
            model.load_weights('model/emotion_detection.h5')
			
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
#            callbacks=[
#                keras.callbacks.LearningRateScheduler(steplr, verbose=1),
#                keras.callbacks.ModelCheckpoint('vgg13-baseline.h5', 
#                monitor='val_acc', 
#                verbose=1,
#                save_best_only=True)],
            validation_data=validation_generator,
            validation_steps=num_val // batch_size
            )
    print(datetime.datetime.now())
    model.save_weights('model/emotion_detection_vgg13.h5')
    plot_model_history(model_info)
	
# emotions will be displayed on your face from the webcam feed
elif mode == "detect":
    model.load_weights('model/emotion_detection_vgg13.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

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
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        #cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        cv2.imshow('Video',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
