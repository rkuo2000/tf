import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

def show_train_history(hisData,train,test): 
    plt.plot(hisData.history[train])
    plt.plot(hisData.history[test])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Load Data
mnist = keras.datasets.mnist # MNIST datasets

# Prepare Data
(x_train_data, y_train),(x_test_data,y_test) = mnist.load_data()
x_train, x_test = x_train_data / 255.0, x_test_data / 255.0

num_classes = 10
img_rows, img_cols = 28, 28 # input image dimensions

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# Build Model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
#optimizer(優化器,可以放入如梯度下降等參數), 
#metric(評估函數,用於計算目前訓練的效果,並不會參與訓練)
#loss(損失函數,用來計算需要調整的方向)
model.summary() #顯示目前建立的模型結構

# Train Model
epochs = 12 
batch_size = 128

train_history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))                          

# Evaluate Model
score = model.evaluate(x_train, y_train) #以訓練資料評估正確率
print ('\nTrain Accuracy:', score[1]) 
score = model.evaluate(x_test, y_test)   #以測試資料評估正確率
print ('\nTest  Accuracy:', score[1])

# Save Model
model.save('model/mnist_cnn.h5')

# Show Train History
show_train_history(train_history,'acc','val_acc') #顯示訓練與測試資料的正確率
show_train_history(train_history,'loss','val_loss') #顯示訓練與測試資料的損失率
