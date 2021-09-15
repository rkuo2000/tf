### download Taiwan Weather Radar (.png)
# usage: python download_weather_radar.py 2020-01-18
import os
import sys
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
import urllib.request
import cv2

date_txt = sys.argv[1] # 2020-01-18
data_path = "datasets/weather/radar/"+date_txt+'/'

if not os.path.exists(data_path):
    os.makedirs(data_path)

# Taiwan Weather Radar "https://www.cwb.gov.tw/Data/radar/CV1_3600_202001180800.png"
weather_radar_url = "https://www.cwb.gov.tw/Data/radar/CV1_3600_"

date_txt = date_txt.replace("-","")

for h in range (24):
    if h<10: 
        h_txt = "0"+str(h)
    else:
        h_txt = str(h)
    for m in range(0,60,10):
        if m==0: 
            m_txt="00"
        else:
            m_txt=str(m)
        file_name = date_txt+h_txt+m_txt+".png"
        imgURL = weather_radar_url+file_name
        print(imgURL)
        try: 
            urllib.request.urlretrieve(imgURL, data_path+file_name)
        except: # download previous radar image if the current image file does not exist
            print("file can not be downloaded !!!")
