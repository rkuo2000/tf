## pip install tf-bodypix
## pip install tfjs-graph-converter 
import numpy as np
import cv2
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    print(frame.shape)
	
	# flip for mirror
    frame = cv2.flip(frame, 1) # 0: flip vertically, 1: flip horizontally
	
    cv2.imshow('CAM', frame)
	
    image_array = tf.keras.preprocessing.image.img_to_array(frame)
    result = bodypix_model.predict_single(image_array)
    mask = result.get_mask(threshold=0.75)
    #tf.keras.preprocessing.image.save_img('output-mask.jpg',mask)

    colored_mask = result.get_colored_part_mask(mask)
    #tf.keras.preprocessing.image.save_img('output-colored-mask.jpg',colored_mask)
	
    c_mask = np.array(colored_mask, dtype=np.uint8)	
    cv2.imshow('Colored Mask', c_mask)

    key=cv2.waitKey(100)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
