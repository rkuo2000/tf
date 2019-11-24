### Transfer Learning example using Keras and Mobilenet V2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

tf.reset_default_graph()
tf.keras.backend.clear_session()
## for GPU
config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

# only train two categories : blue-tit, crow
#prediction_dict = {0: "blue-tit", 1: "crow"}
prediction_dict = {0: "cabbage worm", 1: "corn earworm", 2: "cutworm", 3: "fall armyworm"}
 
# Load Model (MobieNet V2)
model=load_model('model/tl_mobilenetv2.h5')
#model.summary()

def prepare_image(file):
   img_path = 'image/'
   img = image.load_img(img_path + file, target_size=(224, 224))
   img_array = image.img_to_array(img)
   img_array_expanded_dims = np.expand_dims(img_array, axis=0)
   return preprocess_input(img_array_expanded_dims)
	
# Test Model
preprocessed_image = prepare_image('worm.jpg') #crow.jpg labrador1.jpg
predictions = model.predict(preprocessed_image)
print(predictions[0])
maxindex = int(np.argmax(predictions))
print(predictions[0][maxindex],prediction_dict[maxindex])
