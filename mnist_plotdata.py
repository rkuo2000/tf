import random
import matplotlib.pyplot as plt
import tensorflow.keras as keras

mnist = keras.datasets.mnist # MNIST datasets

# Load Data and splitted to train & test sets
# x : the handwritten data, y : the number
(x_train_data, y_train_data), (x_test_data, y_test_data) = mnist.load_data()

print('x_train_data shape:', x_train_data.shape)
print(x_train_data.shape[0], 'train samples')
print(x_test_data.shape[0],  'test samples')

# Print one train_data
print(x_train_data[0])

# Plot one train_data
i = random.randint(0,x_train_data.shape[0])
print(y_train_data[i])
plt.title('Train Data')
plt.imshow(x_train_data[i],cmap='binary')
plt.show()

# Plot one test_data
i = random.randint(0,x_test_data.shape[0])
print(y_test_data[i])
plt.title('Test Data')
plt.imshow(x_test_data[i],cmap='binary')
plt.show()
