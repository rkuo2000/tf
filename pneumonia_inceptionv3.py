### Transfer Learning - Full Model with InceptionV3
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

tf.reset_default_graph()
tf.keras.backend.clear_session()
## for GPU
config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

num_classes = 2 # number of folders under data/chest_xray
target_size = (299,299)

# Dataset Chest_Xray_Pnenumonia
train_dir = 'data/chest_xray/train'
val_dir   = 'data/chest_xray/val'

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
    shuffle=True)
	
validation_datagen = ImageDataGenerator(rescale=rescale)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=target_size,
    class_mode='categorical',
    batch_size=8,
    shuffle = False)

# Load Model (Inception V3)
base_model=keras.applications.inception_v3.InceptionV3(input_shape=(299,299,3), weights='imagenet',include_top=False) 
x=base_model.output
x=GlobalAveragePooling2D()(x)
preds=Dense(num_classes,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)

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
model.save('model/pneumonia_inceptionv3.h5')
