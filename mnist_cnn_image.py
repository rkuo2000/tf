import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

# Load Data
img = cv2.imread('images/digit.jpg', 0)  # read .jpg in gray
x_test = img / 255.0
x_test = x_test.reshape(-1,28,28,1)

# Load Model
model = models.load_model('models/mnist_cnn.h5')

# Test Model
pred = model.predict(x_test)

print("Predict : \t", np.argmax(pred[0]))

plt.imshow(img, cmap='gray')
plt.title('Predict = '+str(np.argmax(pred[0])), color='red')
plt.show()
