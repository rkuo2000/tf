### download Taiwan Weather Satellite (.jpg)
# usage: python download_weather_satellite.py 2020-01-18-00-00
import os
import sys
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import urllib.request
import cv2

date_txt = sys.argv[1] # 2020-01-18-00-00
data_path = "datasets/weather/satellite/"+date_txt[:10]+"/"

if not os.path.exists(data_path):
    os.makedirs(data_path)

# Taiwan weather satellite (TRGB) https://www.cwb.gov.tw/Data/satellite/TWI_VIS_TRGB_1375/TWI_VIS_TRGB_1375-2020-01-14-07-00.jpg
satellite_trgb_url = "https://www.cwb.gov.tw/Data/satellite/TWI_VIS_TRGB_1375/TWI_VIS_TRGB_1375"

file_name = date_txt +".jpg"
imgURL = satellite_trgb_url+"-"+file_name
print(imgURL)
try: 
    urllib.request.urlretrieve(imgURL, data_path+file_name)		
except: # download previous satellite image if the current image file does not exist
    print("file can not be downloaded !!!")
