import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

# Generate some data and noise
# create a matrix of row:300 col:1, value = -1~1
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)

# Generate Y with noise for Neural Network to learn
y_data = np.square(x_data) - 0.5 + noise

# Build Model
model = keras.models.Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(1, activation=None))
model.summary()

# Optimizer : SGD, RMSpro, Adagrad, Adaelta, Adam, Adamax, Nadam
model.compile(optimizer='sgd', loss='mse')

# Train Model
model.fit(x_data, y_data, batch_size=50, epochs=1000)
 
# Plot Result
y_pred = model.predict(x_data)
y_ideal= np.square(x_data) - 0.5
plt.scatter(x_data, y_data)
plt.plot(x_data, y_ideal, 'yellow', lw=2)
plt.plot(x_data, y_pred, 'red', lw=1)
plt.show()
