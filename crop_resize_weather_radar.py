### crop weather radar image
import os, glob
import cv2

os.chdir("dataset/weather/radar/")
data_path = "../../weather/radar_cropped/"

if not os.path.exists(data_path):
    os.makedirs(data_path)
	
top, bottom, left, right = 247, 2337, 772, 2957

for file in glob.glob("*.png"):
    print(file)
    img = cv2.imread(file)
    img = img[top:bottom, left:right]
    img = cv2.resize(img,(300,300))	
    cv2.imwrite(data_path+file, img)
