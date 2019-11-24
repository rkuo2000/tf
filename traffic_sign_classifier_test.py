import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

## for GPU
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

test_file = 'data/GTSRB/test.p'
test=pickle.load(open(test_file,"rb"))
X_test_data,  Y_test_data  = test['data'], test['labels']

# Load Data
X_test = X_test_data / 255.0

# Load Labels
df = pd.read_csv('signnames.csv', header=None)
labels=df.iloc[1:]
labels=labels[1]
labels=labels.reset_index(drop=True)

# Load Model
model = keras.models.load_model('model/traffic_sign.h5')

# Test Model
classes = model.predict_classes(X_test, batch_size=128)

# Plot Prediction
num_test = 5
for x in range(num_test):
    i = random.randint(0,X_test_data.shape[0]) #randomly pick
    plt.subplot(num_test, 1, x+1) 
    plt.imshow(X_test_data[i])
    plt.text(30,0, labels[classes[i]], color='red', fontsize='small')
    plt.axis('off')
    print("Predict : \t", labels[classes[i]])
    print("\n")
plt.show()