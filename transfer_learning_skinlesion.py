### Transfer Learning example using Keras and Mobilenet V2
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# prepare Dataset
batch_size = 32

train_dir = 'dataset/ham10000/train'
val_dir   = 'dataset/ham10000/val'
# Data Generator
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator=train_datagen.flow_from_directory('./dataset/ham10000/train',
        target_size=(224,224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
												
val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator=val_datagen.flow_from_directory('./dataset/ham10000/val',
        target_size=(224,224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)												

num_classes = 7 

# use MobieNet V2 as base model
base_model=keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3),weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

# add Fully-Connected Layers to Model
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) # FC layer 1
x=Dense(64,activation='relu')(x)   # FC layer 2
preds=Dense(num_classes,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

# Check layers no. & name
for i,layer in enumerate(model.layers):
    print(i,layer.name)
	
# set extra layers to trainable (layer #155~159)
for layer in model.layers[:155]:
    layer.trainable=False
for layer in model.layers[155:]:
    layer.trainable=True

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

# Train Model (target is loss <0.01)
num_epochs=20
step_size_train=train_generator.n//train_generator.batch_size
step_size_val=val_generator.n//val_generator.batch_size
model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=num_epochs,
                    validation_data=val_generator, validation_steps=step_size_val)

# Save Model
model.save('model/tl_skinlesion.h5')
