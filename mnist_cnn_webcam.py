import cv2
import tensorflow as tf
import tensorflow.keras as keras

## for GPU
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

model = keras.models.load_model('model/mnist_cnn.h5')

cap = cv2.VideoCapture(0)

while True:
	ret, img = cap.read()
	x,y,w,h  = 200,200,200,200		
	roi = img[x:x+w, y:y+h]	
	roi_gray= cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)		
	ret,roi_thresh = cv2.threshold(roi_gray,130,255,cv2.THRESH_BINARY_INV)
	roi_thresh = cv2.dilate(roi_thresh, (3, 3))
	roi_small = cv2.resize(roi_thresh, (28, 28), interpolation=cv2.INTER_CUBIC)
	x_test=roi_small/255.0
	x_test=x_test.reshape(-1,28,28,1)
	predict=model.predict_classes(x_test)
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)	
	cv2.putText(img, str(int(predict[0])), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
	cv2.imshow('Cam', img)
	cv2.imshow('Thresh', roi_thresh)
	cv2.waitKey(1)		

cap.release()
cv2.destroyAllWindows()
