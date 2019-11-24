### Transfer Learning example using Keras and Mobilenet V2
# Download Dataset
# $python download_google_images.py "blue tit"
# $python download_google_images.py crow
# Please remove improper image files in downloads folder !!!
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

## for GPU
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

num_classes = 2 # number of folders under ./downloads
# Data Generator
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator=train_datagen.flow_from_directory('./downloads',
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                batch_size=32,
                                                class_mode='categorical',
                                                shuffle=True)

# Load Model (MobieNet V2)
base_model=keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3), weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
base_model.trainable = False

# Add Extra Layers to Model
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) 
x=Dense(512,activation='relu')(x) 
preds=Dense(num_classes,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)

# Compile Model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# Train Model (target is loss <0.01)
num_epochs=80
step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=num_epochs)

# Save Model
model.save('model/tl_mobilenetv2.h5')