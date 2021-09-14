### crop weather satellite image
import os, glob
import cv2

os.chdir("dataset/weather/satellite/")
data_path = "../../weather/satellite_cropped/"

if not os.path.exists(data_path):
    os.makedirs(data_path)
	
top, bottom, left, right = 45, 1318, 80, 1303

for file in glob.glob("*.jpg"):
    print(file)
    img = cv2.imread(file)
    img = img[top:bottom, left:right]
    img = cv2.resize(img,(300,300))
    cv2.imwrite(data_path+file, img)
