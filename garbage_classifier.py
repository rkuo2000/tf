### Garbage Classifier - Train
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout
from tensorflow.keras.models  import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_path = 'dataset/garbage'

# Data Generator
batch_size = 32
train=ImageDataGenerator(horizontal_flip=True, vertical_flip=True,validation_split=0.1,rescale=1./255,
                         shear_range = 0.1,zoom_range = 0.1,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,)
test=ImageDataGenerator(rescale=1./255,validation_split=0.1)

train_generator=train.flow_from_directory(data_path,target_size=(300,300),batch_size=batch_size,
                                          class_mode='categorical',subset='training')
test_generator=test.flow_from_directory(data_path,target_size=(300,300),batch_size=batch_size,
                                        class_mode='categorical',subset='validation')
										
										
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)

num_classes = 6

# Build Model
model = Sequential()
   
model.add(Conv2D(32,(3,3), padding='same', input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.summary()

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
model_path="model/garbage_cnn.h5"
checkpoint1 = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]

num_epochs = 100
step_size_train=train_generator.n//train_generator.batch_size #2276/32
step_size_test =test_generator.n//test_generator.batch_size   #251/32

history = model.fit_generator(train_generator, epochs=num_epochs, steps_per_epoch=step_size_train,validation_data=test_generator,
                    validation_steps=step_size_test,callbacks=callbacks_list)