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
	
	## bodypix
    image_array = tf.keras.preprocessing.image.img_to_array(frame)
    result = bodypix_model.predict_single(image_array)
    mask = result.get_mask(threshold=0.75)
    #tf.keras.preprocessing.image.save_img('output-mask.jpg',mask)

    colored_mask = result.get_colored_part_mask(mask)
    #tf.keras.preprocessing.image.save_img('output-colored-mask.jpg',colored_mask)

    ## convert to uint8 np.array
    #c_mask = colored_mask.astype(np.uint8)	
    c_mask = np.array(colored_mask, dtype=np.uint8)	
    cv2.imshow('Colored Mask', c_mask)

    hsv = cv2.cvtColor(c_mask, cv2.COLOR_RGB2HSV)
	
    ## find color range of upper-body 
    #lower = np.array([100, 200, 80], dtype="uint8")
    #upper = np.array([200, 255, 128], dtype="uint8")
    #upperbody_mask = cv2.inRange(c_mask, lower, upper)
    #cv2.imshow('upperbody mask', upperbody_mask)	
 
    ## use inRange to generate upper-body mask
    lower = np.array([40, 155, 240], dtype="uint8")
    upper = np.array([50, 160, 250], dtype="uint8")
    upperbody_mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('upper mask', upperbody_mask)	

    ## use inRange to generate head mask
    lower = np.array([130, 155, 170], dtype="uint8")
    upper = np.array([145, 170, 180], dtype="uint8")
    head_mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('head mask', head_mask)
	
    key=cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
