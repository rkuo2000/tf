### download Taiwan Weather Radar (.png)
# usage: python download_weather_radar.py 2020-01-18-00-20
import os
import sys
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import urllib.request
import cv2

date_txt = sys.argv[1] # 2020-01-18
print(date_txt[:10])
data_path = "datasets/weather/radar/"+date_txt[:10]+'/'

if not os.path.exists(data_path):
    os.makedirs(data_path)

# Taiwan Weather Radar "https://www.cwb.gov.tw/Data/radar/CV1_3600_202001180800.png"
weather_radar_url = "https://www.cwb.gov.tw/Data/radar/CV1_3600_"

date_txt = date_txt.replace("-","")

file_name = date_txt+".png"
imgURL = weather_radar_url+file_name
print(imgURL)
try: 
    urllib.request.urlretrieve(imgURL, data_path+file_name)
except: 
    # download previous radar image if the current image file does not exist
    print("file can not be downloaded !!!")
