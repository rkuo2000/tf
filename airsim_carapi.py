# AirSim Car API (control car throttle, steering, and capture a image)

# $pip install airsim
import airsim 
import numpy as np
import time
import matplotlib.pyplot as plt

# connect to the AirSim simulator
client = airsim.CarClient() # for QuadCopter use MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

# get state of the car
car_state = client.getCarState()
print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

# set the controls for car
car_controls.throttle = 1
car_controls.steering = 1
client.setCarControls(car_controls)

# let car drive a bit
time.sleep(6) 

# get another car state
car_state = client.getCarState()
print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

# stop the car
car_controls.throttle = 0
car_controls.steering = 0
client.setCarControls(car_controls)

# get camera image
responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
response = responses[0]
img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
img_rgb = img1d.reshape(response.height, response.width, 3)
print(img_rgb.shape)

# show camera image
plt.imshow(img_rgb)
plt.show()	


