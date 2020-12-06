import sys
import cv2
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

if len(sys.argv)>1:
    file = sys.argv[1]
else:
    file = "image/persons.jpg"

bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

image = tf.keras.preprocessing.image.load_img(file)
image.show() # PIL Image

image_array = tf.keras.preprocessing.image.img_to_array(image)
result = bodypix_model.predict_single(image_array)

mask = result.get_mask(threshold=0.75)
tf.keras.preprocessing.image.save_img('bodypix-mask.jpg',mask)
print(type(mask))

colored_mask = result.get_colored_part_mask(mask)
tf.keras.preprocessing.image.save_img('bodypix-colored-mask.jpg',colored_mask)
print(type(colored_mask))


import matplotlib.pyplot as plt
plt.imshow(colored_mask)
plt.show()
