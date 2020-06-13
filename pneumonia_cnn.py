### Pneumonia Detection using CNN
import os
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

## for GPU
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

#def dir_file_count(directory):
#    return sum([len(files) for r, d, files in os.walk(directory)])
	
num_classes = 2 # number of folders under data/chest_xray
target_size = (224,224)

# Dataset Chest_Xray_Pnenumonia
train_dir = 'dataset/chest_xray/train'
val_dir   = 'dataset/chest_xray/val'

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
    batch_size=32,
    color_mode="grayscale",	
    shuffle=True)
	
validation_datagen = ImageDataGenerator(rescale=rescale)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=target_size,
    class_mode='categorical',
    batch_size=8,
    color_mode="grayscale",
    shuffle = False)
		
# Build Model
model = keras.models.Sequential()
# block 1
model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(224,224,1)))
model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# block 2
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# block 3
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# block 4
model.add(Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# block 5
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# fully-connected layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes , activation='softmax'))

# Compile Model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# Train Model 
num_epochs=100

model.fit_generator(generator=train_generator, 
	steps_per_epoch=train_generator.n // train_generator.batch_size, 
	epochs=num_epochs,
	validation_data=validation_generator, 
	validation_steps=validation_generator.n // validation_generator.batch_size)
			
# Save Model
model.save('model/pneumonia_cnn.h5')
