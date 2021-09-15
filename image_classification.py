# Image Classification
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, resnet50, inception_v3 

# Load Model
model = vgg16.VGG16(weights='imagenet')

# Load test data
img_path = 'images/cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = vgg16.preprocess_input(x)

# Model Predict
preds = model.predict(x)

# Print Prediction
# decode the results into a list of tuples (class, description, probability)
dec_preds =  vgg16.decode_predictions(preds, top=3)[0]
print('Predicted:', dec_preds)
for item in dec_preds:
    print(item[1], item[2])
