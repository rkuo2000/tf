import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

## for GPU
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

mnist = keras.datasets.mnist # MNIST datasets

# Load Data
(x_train_data, y_train),(x_test_data,y_test) = mnist.load_data()
x_train, x_test = x_train_data / 255.0, x_test_data / 255.0

x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# Load Model
model = keras.models.load_model('model/mnist_cnn.h5')

# Test Model
classes = model.predict_classes(x_test, batch_size=128)

# Plot Prediction
for x in range(3):
    i = random.randint(0,x_test.shape[0]) #randomly pick        
    plt.imshow(x_test_data[i],cmap='binary')
    plt.title('Predict = '+str(classes[i]), color='red')
    plt.show()
    print("Predict : \t", classes[i])
    print("\n")
