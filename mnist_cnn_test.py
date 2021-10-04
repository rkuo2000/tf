import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras import models

mnist = datasets.mnist # MNIST datasets

# Load Data
(x_train_data, y_train),(x_test_data,y_test) = mnist.load_data()
x_train, x_test = x_train_data / 255.0, x_test_data / 255.0

x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# Load Model
model = models.load_model('models/mnist_cnn.h5')

# Test Model
pred = model.predict(x_test)

# Plot Prediction
for x in range(3):
    i = random.randint(0,x_test.shape[0]) #randomly pick        
    plt.imshow(x_test_data[i],cmap='binary')
    plt.title('Predict = '+str(np.argmax(pred[i])), color='red')
    plt.show()
    print("Predict : \t", np.argmax(pred[i]))
    print("\n")
