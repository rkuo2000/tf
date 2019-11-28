import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

# Load Data
if (len(sys.argv)>1):
    img = cv2.imread(sys.argv[1])
else:
    img = cv2.imread('image/gesture.jpg')

img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)

x_data = img / 255.0
print(x_data.shape)
x_data = img.reshape(1,224,224,3)

# Dictionary
dict = {0: 'paper', 1: 'rock', 2: 'scissors'}

# Load Model
model = load_model('model/gesture_cnn.h5')

# Test Model
predictions = model.predict(x_data)
maxindex = int(np.argmax(predictions))
print(predictions[0][maxindex], dict[maxindex])

# Display Image       
cv2.imshow('Image', img)
	
cv2.waitKey(0)
cv2.destroyAllWindows()