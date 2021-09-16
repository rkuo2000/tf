import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing

from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

symbol = sys.argv[1] # 'GOOGL' or 'MSFT'

history_points = 50

csv_path = 'datasets/stocks/'+symbol+'.csv'

# Read Dataset & Normalise it
def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    ## reverse index because .csv top column is most recent price 
    data = data[::-1]
    data = data.reset_index()
    data = data.drop('index', axis=1)
    print(data)

    # normaliser
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data) 
    # using the last {history_points} open high low close volume data points, predict the next open value
    ohlcv_histories_normalised =      np.array([data_normalised[i  : i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    print(ohlcv_histories_normalised.shape)    
    next_day_open_values_normalised = np.array([data_normalised[:,0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])   

    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data.loc[:,"1. open"][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)
    
    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)
    
    print(ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0])
    return ohlcv_histories_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

# Read Dataset
ohlcv_histories, next_day_open_values, unscaled_y, y_scaler = csv_to_dataset(csv_path)

# splitting the dataset up into train and test sets
test_split = 0.9 # 90% stock-history for training, most-recent 10% stock-history for testing
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

print(ohlcv_train.shape)

# Build Model (RNN)
lstm_input = layers.Input(shape=(history_points, 5), name='lstm_input')

x = layers.LSTM(50, name='lstm_0')(lstm_input)
x = layers.Dropout(0.2, name='lstm_dropout_0')(x)
x = layers.Dense(64, name='dense_0')(x)
x = layers.Activation('sigmoid', name='sigmoid_0')(x)
x = layers.Dense(1, name='dense_1')(x)
output = layers.Activation('linear', name='linear_output')(x)
model = models.Model(inputs=lstm_input, outputs=output)

model.summary()

# Compile Model
model.compile(loss='mse', optimizer='adam')

# Train Model
num_epochs = 50
batch_size = 32
model.fit(x=ohlcv_train, y=y_train, batch_size=batch_size, epochs=num_epochs, shuffle=True, validation_split=0.1)

# Evaluate Model
evaluation = model.evaluate(ohlcv_test, y_test)
print(evaluation)

y_test_predicted = model.predict(ohlcv_test)

# model.predict returns normalised values, now we scale them back up using the y_scaler from before
y_test_predicted = y_scaler.inverse_transform(y_test_predicted)

# also getting predictions for the entire dataset, just to see how it performs
y_predicted = model.predict(ohlcv_histories)
y_predicted = y_scaler.inverse_transform(y_predicted)
print(y_predicted.shape)

print(unscaled_y_test.shape == y_test_predicted.shape)
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)

# Plot stock prediction
plt.gcf().set_size_inches(22, 15, forward=True)
start = 0
end = -1
real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')
plt.title('symbol = '+symbol)
plt.legend(['Real', 'Predicted'])
plt.show()
