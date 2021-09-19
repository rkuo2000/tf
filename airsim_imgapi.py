# AirSim Car - Image API

# $pip install airsim
import airsim 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# for QuadCopter use MultirotorClient()
client = airsim.CarClient() 

responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
response = responses[0]
img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
img_rgb = img1d.reshape(response.height, response.width, 3)

print(img_rgb.shape)
plt.imshow(img_rgb)
plt.show()
