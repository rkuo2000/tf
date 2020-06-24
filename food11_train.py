# USAGE : python food11_train.py
import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
import efficientnet.tfkeras as efn

print(os.listdir('../input/food11'))
trainPath = 'dataset/Food-11/training'
validPath = 'dataset/Food-11/validation'
testPath  = 'dataset/Food-11/evaluation'

## Data Augmentation
target_size = (224,224)
batch_size  = 16

# Training Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,    
    horizontal_flip=True,
    fill_mode="nearest")

train_generator = train_datagen.flow_from_directory(
    trainPath,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',    
    shuffle=True,
    seed=42,
    class_mode='categorical')

# Validation Data
valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_directory(
    validPath,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical')

# Test Data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    testPath,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',    
    class_mode='categorical')
	
num_train = 9866
num_valid = 3430
num_test  = 3347
num_classes = 11
input_shape = (224,224,3)

## Build Model
net = efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)

# add two FC layers (with L2 regularization)
x = net.output
x = GlobalAveragePooling2D()(x)

x = Dense(256)(x)
x = Dense(32)(x)

# Output layer
out = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=net.input, outputs=out)
model.summary()

# Compile Model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

## Train Model
num_epochs = 20

# Train Model
history = model.fit_generator(train_generator,
    steps_per_epoch=num_train // batch_size,
    epochs=num_epochs, 
    validation_data=valid_generator, 
    validation_steps=num_valid // batch_size)
	#, callbacks=[checkpoint])

## Save Model
model.save('model/food11.h5')

## Evaluate Model
score = model.evaluate(valid_generator)
print(score)

## Confusion Matrix report

# Validation's Confusion Matrix
predY=model.predict(valid_generator)
y_pred = np.argmax(predY,axis=1)
y_actual = valid_generator.classes
cm = confusion_matrix(y_actual, y_pred)
print(cm)

# report Confusion Matrix
print(classification_report(y_actual, y_pred, target_names=labels))

# Test's Confusion Matrix
predY=model.predict(test_generator)
y_pred = np.argmax(predY,axis=1)
y_actual = test_generator.classes
cm = confusion_matrix(y_actual, y_pred)
print(cm)

# report Confusion Matrix
print(classification_report(y_actual, y_pred, target_names=labels))
