# Transfer Learning Detection using webcam
import numpy as np
import cv2
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

# only train two categories
prediction_dict = {0: "blue-tit", 1: "crow"}

# Load Model (MobieNet V2)
model=load_model('model/tl_mobilenetv2.h5')
#model.summary()

cap = cv2.VideoCapture(0)

# Test Model
while True:
	ret, img = cap.read()
	x,y,w,h  = 200, 100, 224,224		
	roi = img[x:x+w, y:y+h]	
	img_array = image.img_to_array(roi)
	img_array_expanded_dims = np.expand_dims(img_array, axis=0)
	preprocessed_image = preprocess_input(img_array_expanded_dims)
	predictions = model.predict(preprocessed_image)	
	maxindex = int(np.argmax(predictions))
	#print(predictions[0])
	#print(predictions[0][maxindex],prediction_dict[maxindex])
	
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)	
	cv2.putText(img, prediction_dict[maxindex], (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
	cv2.imshow('Cam', img)
    # Press 'q' to quit
	if cv2.waitKey(1) == ord('q'):
		break	

cap.release()
cv2.destroyAllWindows()