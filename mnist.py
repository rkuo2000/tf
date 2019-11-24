import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Load Data
mnist = keras.datasets.mnist # MNIST datasets

# Data split into two sets : Train and Test
# x is the hand-written pixel data, y is its number
(x_train_data, y_train),(x_test_data,y_test) = mnist.load_data()
# normalize the data between 0 and 1 for training
x_train, x_test = x_train_data / 255.0, x_test_data / 255.0 

# Build Model
model = keras.models.Sequential([
Flatten(input_shape=(28,28)),
Dense(512, activation='relu'),
Dropout(0.5),
Dense(10, activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

# Train Model                    
model.fit(x_train, y_train, batch_size=128, epochs=1) 

# Evaluate Model
score = model.evaluate(x_test, y_test)
print('Test  Accuracy:', score[1])

# Save Model
model.save('model/mnist.h5')
