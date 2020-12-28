## pip install tf-bodypix
## pip install tfjs-graph-converter
import sys
import cv2
import numpy as np
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

if len(sys.argv)>1:
    file = sys.argv[1]
else:
    file = "image/persons.jpg"

img = cv2.imread(file)
print(type(img))
print(img.shape)

bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

# load image file & display it
image = tf.keras.preprocessing.image.load_img(file)
#image.show() # PIL Image

image_array = tf.keras.preprocessing.image.img_to_array(image)
print(type(image_array))
image = np.array(image_array, dtype=np.uint8)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('image',image)
	
result = bodypix_model.predict_single(image_array)

mask = result.get_mask(threshold=0.75)
#tf.keras.preprocessing.image.save_img('bodypix-mask.jpg',mask)
print(type(mask))

colored_mask = result.get_colored_part_mask(mask)
#tf.keras.preprocessing.image.save_img('bodypix-colored-mask.jpg',colored_mask)
colored_mask = colored_mask.astype(np.uint8)
cv2.imshow('Colored Mask', colored_mask)
cv2.imwrite('colored_mask.jpg', colored_mask)

# get colored facemask
colored_facemask = result.get_colored_part_mask(mask, part_names=['left_face', 'right_face'])
colored_facemask = colored_facemask.astype(np.uint8)
cv2.imshow('Colored Face Mask', colored_facemask)
cv2.imwrite('colored_facemask.jpg', colored_facemask)

# convert colored facemask to bw facemask
gray_facemask = cv2.cvtColor(colored_facemask, cv2.COLOR_BGR2GRAY)
(thresh, facemask) = cv2.threshold(gray_facemask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('Face Mask', facemask)
cv2.imwrite('facemask.jpg', facemask)

## find contours and its bounding box
contours, hierarchy = cv2.findContours(facemask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
print("--- bounding box of head ---")
print(bounding_boxes)

## show faces only
frame = cv2.bitwise_and(image,image,mask = facemask) 
cv2.imshow('faces only', frame)

## draw bounding box on head
if len(bounding_boxes)>0:
    for x,y,w,h in bounding_boxes:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)	
cv2.imshow('faces bbox', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
