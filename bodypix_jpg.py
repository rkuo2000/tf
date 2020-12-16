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
tf.keras.preprocessing.image.save_img('bodypix-mask.jpg',mask)
print(type(mask))

colored_mask = result.get_colored_part_mask(mask)
tf.keras.preprocessing.image.save_img('bodypix-colored-mask.jpg',colored_mask)

## show colored mask
#import matplotlib.pyplot as plt
#plt.imshow(colored_mask)
#plt.show()
colored_mask = colored_mask.astype(np.uint8)
cv2.imshow('Colored Mask', colored_mask)
cv2.imwrite('colored_mask.jpg', colored_mask)

## convert RGB to HSV
hsv = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2HSV)
print(hsv.shape)

## use Paint tool to identify pixel coordinates, then print pixel color
#print("--- upper body ---")
#print(hsv[270,180]) # [y,x]
#print(hsv[240,525]) # [y,x]
#print(hsv[416,560]) # [y,x]

## use inRange to generate upper-body mask
lower = np.array([40, 155, 240], dtype="uint8")
upper = np.array([50, 160, 250], dtype="uint8")
upperbody_mask = cv2.inRange(hsv, lower, upper)
#
cv2.imshow('uppder-body mask', upperbody_mask)
cv2.imwrite('tryon_mask.jpg', upperbody_mask)

## use inRange to generate head mask
lower = np.array([130, 155, 170], dtype="uint8")
upper = np.array([145, 170, 180], dtype="uint8")
head_mask = cv2.inRange(hsv, lower, upper)
#
cv2.imshow('head mask', head_mask)
cv2.imwrite('head_mask.jpg', head_mask)

## find contours and its bounding box
contours, hierarchy = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
print("--- bounding box of head ---")
print(bounding_boxes)

## show head only
frame = cv2.bitwise_and(image,image,mask = head_mask) 
cv2.imshow('head only', frame)

## draw bounding box on head
if len(bounding_boxes)>0:
    for x,y,w,h in bounding_boxes:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)	
cv2.imshow('head bbox', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
