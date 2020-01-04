import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load Data
mnist = keras.datasets.mnist # MNIST dataset
(x_train_data, y_train),(x_test_data,y_test) = mnist.load_data()
x_train, x_test = x_train_data / 255.0, x_test_data / 255.0 

x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# Build Model
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), padding='same', activation='tanh', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (5, 5), padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()

# Train Model                    
model.fit(x_train, y_train, batch_size=128, epochs=12)

# Save Model
model.save('model/mnist_lenet.h5')
print('model/mnist_lenet.h5 is saved')

# Evaluate Model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test  Accuracy:', score[1])
