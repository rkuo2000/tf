
from tensorflow.keras import models, layers, datasets
import matplotlib.pyplot as plt

# Load Data
mnist = datasets.mnist # MNIST datasets
(x_train_data, y_train),(x_test_data,y_test) = mnist.load_data()

x_train, x_test = x_train_data / 255.0, x_test_data / 255.0
 
num_classes = 10 #0~9
img_rows, img_cols = 28, 28 # input image dimensions

print('x_train shape:', x_train.shape)
print('train samples:', x_train.shape[0])
print('test samples:', x_test.shape[0])

x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# Build Model
model = models.Sequential()
model.add(layers, Conv2D(32, kernel_size=(5, 5),activation='relu,'padding='same',input_shape=(28,28,1)))
model.add(layers, MaxPooling2D(pool_size=(2, 2)))
model.add(layers, Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(layers, MaxPooling2D(pool_size=(2, 2)))
model.add(layers, Dropout(0.25))
model.add(layers, Flatten())
model.add(layers, Dense(128, activation='relu'))
model.add(layers, Dropout(0.5))
model.add(layers, Dense(num_classes, activation='softmax'))
model.summary() 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# Train Model
epochs = 12 
batch_size = 128

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))                          

# Save Model
models.save_model(model, 'models/mnist_cnn.h5')

# Evaluate Model
score = model.evaluate(x_train, y_train, verbose=0) 
print('\nTrain Accuracy:', score[1]) 
score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest  Accuracy:', score[1])
print()

# Show Train History
keys=history.history.keys()
print(keys)

def show_train_history(hisData,train,test): 
    plt.plot(hisData.history[train])
    plt.plot(hisData.history[test])
    plt.title('Training History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
	
show_train_history(history, 'loss', 'val_loss')
show_train_history(history, 'accuracy', 'val_accuracy')
