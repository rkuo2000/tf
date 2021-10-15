### Gesture Detection using CNN
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
				
num_classes = 5 # number of gestures
target_size = (96,96)

# Dataset 
train_dir = 'datasets/gesture'

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
    color_mode="grayscale",	
    shuffle=True)
	
# Build Model
model = models.Sequential()
# conv-layer 1
model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(96,96,1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# conv-layer 2
model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# conv-layer 3
model.add(layers.Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# conv-layer 4
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# fully-connected layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes , activation='softmax'))

model.summary()

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# Train Model 
num_epochs=100
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=num_epochs)
			
# Save Model
models.save_model(model, 'models/gesture_cnn.h5')
