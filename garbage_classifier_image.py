### Garbage Classifier - Predict
import numpy as np
import cv2

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

model = load_model('model/garbage.h5')

pic = cv2.imread('image/garbage.jpg')
pic = cv2.resize(pic, (300, 300))
pic = np.expand_dims(pic, axis=0)
classes = model.predict_classes(pic)
print(classes[0])   
print(labels[classes[0]])
