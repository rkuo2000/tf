### Gesture Detection using CNN
import os
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
				
num_classes = 5 # number of gestures
target_size = (96,96)

# Dataset 
train_dir = 'dataset/gesture'

# Data Generator
rescale = 1./255
train_datagen = ImageDataGenerator(
    rescale=rescale,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    class_mode='categorical',
    batch_size=10,
    color_mode="rgb",	
    shuffle=True)
	
# Build Model
model = Sequential()
# block 1
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(96,96,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# block 2
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# block 3
model.add(Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# block 3
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# fully-connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes , activation='softmax'))

# Compile Model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# Train Model 
num_epochs=100

model.fit_generator(generator=train_generator, 
	steps_per_epoch=train_generator.n // train_generator.batch_size, 
	epochs=num_epochs)
			
# Save Model
model.save('model/gesture_cnn.h5')
