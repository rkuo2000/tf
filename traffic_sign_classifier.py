### Traffic Sign Classifier
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, Dense
from tensorflow.keras.utils import to_categorical

# Load Data
train_file = 'dataset/GTSRB/train.p'
test_file = 'dataset/GTSRB/test.p'
train=pickle.load(open(train_file,"rb"))
test=pickle.load(open(test_file,"rb"))

X_train_data, Y_train_data = train['data'],train['labels']
X_test_data,  Y_test_data  = test['data'],test['labels']

image_shape = X_train_data.shape[1:]
num_classes = len(set(Y_train_data))

print("Training   examples :", X_train_data.shape[0])
print("Test       examples :", X_test_data.shape[0])
print("Image data shape  :", image_shape)
print("number of classes :", num_classes)

# Normalize Data
X_train = X_train_data / 255.0
X_test  = X_test_data  / 255.0

Y_train = to_categorical(Y_train_data, num_classes = num_classes)
Y_test  = to_categorical(Y_test_data,  num_classes = num_classes)

# Split Data
X_train,X_valid,Y_train,Y_valid = train_test_split(X_train,Y_train,test_size = 0.3,random_state=0)

# Build Model
model = keras.models.Sequential()
model.add(Conv2D(32,(3,3), padding='same', input_shape = (32,32,3),activation = 'relu'))
model.add(Conv2D(32,(3,3), padding='same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),padding='same',activation = 'relu'))
model.add(Conv2D(64,(3,3),padding='same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3),padding='same',activation = 'relu'))
model.add(Conv2D(128,(3,3),padding='same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes ,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.summary()

# Train Model
model.fit(X_train,Y_train,validation_data=(X_valid,Y_valid),epochs=30,batch_size=128,verbose =1)

# Evaluate Model
score = model.evaluate(X_test, Y_test)
print ('\nTest Accuracy:', score[1])

# Save Model
model.save('model/traffic_sign.h5')