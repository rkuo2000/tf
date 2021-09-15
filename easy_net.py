# Easy Net
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Generate Data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.1* X + 0.3 + np.random.normal(0, 0.01, (200,))

# Split Data to two sets : Train and Test
X_train, Y_train = X[:160], Y[:160]
X_test,  Y_test  = X[160:], Y[160:]

# Build Model
model = models.Sequential()

# add one fully-connected neuron layer
model.add(layers.Dense(units=1, input_dim=1))

model.summary() 

# Compile Mode 
# select optimizer and set loss function
# optimizer ： SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam 
model.compile(loss='smse', optimizer='sgd',metrics=['accuracy'])

# For Tensorboard
tbCallBack = keras.callbacks.TensorBoard(log_dir='.', histogram_freq=0, write_graph=True, write_images=True)

# Train Model
model.fit(X_train, Y_train, batch_size = 40, epochs=300, callbacks=[tbCallBack])

# Evaluate Model
cost = model.evaluate(X_test, Y_test, batch_size=40)
print("test cost: {}".format(cost))
W, b = model.layers[0].get_weights()
print("weights = {}, biases= {}".format(W, b))

# Plot Prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
