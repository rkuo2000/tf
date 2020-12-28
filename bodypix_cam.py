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

    #colored_mask = result.get_colored_part_mask(mask)
    #colored_mask = np.array(colored_mask, dtype=np.uint8)	
	
	### show face only
    colored_facemask = result.get_colored_part_mask(mask, part_names=['left_face','right_face'])
    colored_facemask = np.array(colored_facemask, dtype=np.uint8)
    gray_facemask = cv2.cvtColor(colored_facemask, cv2.COLOR_BGR2GRAY)
    (thresh, facemask) = cv2.threshold(gray_facemask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    face = cv2.bitwise_and(frame,frame, mask=facemask)
    cv2.imshow('Face', face)
	
	### show torso-front only
    #colored_torsofrontmask = result.get_colored_part_mask(mask, part_names=['torso_front'])
    #colored_torsofrontmask = np.array(colored_torsofrontmask, dtype=np.uint8)
    #gray_torsofrontmask = cv2.cvtColor(colored_torsofrontmask, cv2.COLOR_BGR2GRAY)
    #(thresh, torsofrontmask) = cv2.threshold(gray_torsofrontmask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #torsofront = cv2.bitwise_and(frame,frame, mask=torsofrontmask)
    #cv2.imshow('Torso Front', torsofront)	

    ## convert to uint8 np.array
    #c_mask = colored_mask.astype(np.uint8)	

    key=cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
