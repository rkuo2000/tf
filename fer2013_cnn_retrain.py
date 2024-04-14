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

from tensorflow.keras import models, layers

## Load Model
model = models.load_model("fer2013_cnn.h5")
model.summary()

## Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

## Train Model
history = model.fit(train_generator, validation_data=test_generator, epochs=100)

## Evaluate Model
score = model.evaluate(test_generator) 
print('Test loss: ', score[0])
print('Test accuracy: ', score[1]) 

## Save Model
#models.save_model(model, 'fer2013_cnn.h5') 
model.save('fer2013_cnn.h5')
