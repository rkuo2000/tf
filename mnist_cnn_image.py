import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

# Load Data
img = cv2.imread('image/digit.jpg', 0)  # read .jpg in gray
x_test = img / 255.0
x_test = x_test.reshape(-1,28,28,1)

# Load Model
model = keras.models.load_model('model/mnist_cnn.h5')

# Test Model
classes = model.predict_classes(x_test)

plt.imshow(img, cmap='gray')
plt.title('Predict = '+str(classes[0]), color='red')
plt.show()
print("Predict : \t", classes[0])
