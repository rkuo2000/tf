train_dir = '/home/rkuo/Datasets/FER2013_clean/train'
test_dir  = '/home/rkuo/Datasets/FER2013_clean/test'

## Dataset Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

batch_size = 64
target_size = (48,48)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False)

labels = list(train_generator.class_indices.keys())
print(labels)

## Build Model
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout

input_shape = (48,48,1) # img_rows, img_colums, color_channels
num_classes = len(labels) # 7

## Build Model
model = models.Sequential()
# 1st convolution layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
# 2nd convolution layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
# 3rd convolution layer
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
# 4th convolution layer
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
# fully-connected layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))  
model.add(Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))  
model.add(Dropout(0.2))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

## Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

## Train Model
history = model.fit(train_generator, validation_data=test_generator, epochs=300)

## Evaluate Model
score = model.evaluate(test_generator) 
print('Test loss: ', score[0])
print('Test accuracy: ', score[1]) 

## Save Model
#models.save_model(model, 'fer2013_cnn.h5') 
model.save('fer2013_cnn.h5')

## convolution-layers = 32,64,128,256
## fc-layers = 512,64
## Total params: 1,603,207
## Epochs = 300
## Train accuracy = 97.85%
## Test  accuracy = 62.19% 
