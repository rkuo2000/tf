import random
import matplotlib.pyplot as plt
import tensorflow.keras as keras

mnist = keras.datasets.mnist # MNIST datasets

# Load Data
(x_train_data, y_train),(x_test_data,y_test) = mnist.load_data()
x_train, x_test = x_train_data / 255.0, x_test_data / 255.0

# Load Model
model = keras.models.load_model('model/mnist.h5')

# Test Model
classes = model.predict_classes(x_test, batch_size=128)

# Plot Prediction
num_test = 3
for x in range(num_test):
    i = random.randint(0,x_test.shape[0]) #randomly pick        
    plt.imshow(x_test_data[i],cmap='binary')
    plt.title('Predict = '+str(classes[i]), color='red')
    plt.show()    
    print("Predict : \t", classes[i])
    print("\n")
