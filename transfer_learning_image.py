### Transfer Learning example using Keras and Mobilenet V2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

num_classes = 2
prediction_labels = {0: "blue-tit", 1: "pica-pica"}
# num_classes = 4
#prediction_dict = {0: "cabbage worm", 1: "corn earworm", 2: "cutworm", 3: "fall armyworm"}
 
# Load Model (MobieNet V2)
print('model loading...')
model=load_model('model/tl_birds2.h5')

def prepare_image(file):
   img_path = 'image/'
   img = image.load_img(img_path + file, target_size=(224, 224))
   img_array = image.img_to_array(img)
   img_array_expanded_dims = np.expand_dims(img_array, axis=0)
   return preprocess_input(img_array_expanded_dims)
	
# Test Model
filename = 'blue_tit.jpg'
print('Test Image: '+filename)
preprocessed_image = prepare_image(filename)
predictions = model.predict(preprocessed_image)
print(predictions[0])
maxindex = int(np.argmax(predictions))
print(predictions[0][maxindex],prediction_labels[maxindex])
print()
filename = 'pica_pica.jpg'
print('Test Image: '+filename)
preprocessed_image = prepare_image(filename)
predictions = model.predict(preprocessed_image)
print(predictions[0])
maxindex = int(np.argmax(predictions))
print(predictions[0][maxindex],prediction_labels[maxindex])