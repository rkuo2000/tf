# hand-gesture detection
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

# Dictionary
prediction_dict = {0: "palm", 1: "L", 2: "L", 3: "fist", 4:"thumb", 5:"index", 6:"ok", 7:"palm moved", 8:"C", 9:"Down"}
 
# Load Model 
model=load_model('model/hand-gesture.h5')
model.summary()

def prepare_image(file):
   img_path = 'image/'
#   img = image.load_img(img_path + file, target_size=(150, 150))   
   img = cv2.imread(img_path+file,cv2.IMREAD_GRAYSCALE)
#   img = cv2.resize(img, (150,150))
#   img_array = image.img_to_array(img)
   img_array = np.array(img)
   img_array_expanded_dims = np.expand_dims(img_array, axis=0)
   return preprocess_input(img_array_expanded_dims)
	
# Test Model
preprocessed_image = prepare_image('gesture_gray.jpg') 
predictions = model.predict(preprocessed_image)
print(predictions[0])
maxindex = int(np.argmax(predictions))
print(predictions[0][maxindex],prediction_dict[maxindex])
