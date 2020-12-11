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
image.show() # PIL Image

image_array = tf.keras.preprocessing.image.img_to_array(image)
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

# use Paint tool to identify pixel coordinates, then print pixel color
print(colored_mask.shape)    # [453,628,3]
print(colored_mask[252,182]) # [y,x]
print(colored_mask[211,512]) # [y,x]
print(colored_mask[412,563]) # [y,x]

lower = np.array([100, 200, 80], dtype="uint8")
upper = np.array([200, 255, 128], dtype="uint8")
tryon_mask = cv2.inRange(colored_mask, lower, upper)

cv2.imshow('TryOn', tryon_mask)
cv2.imwrite('tryon_mask.jpg', tryon_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
