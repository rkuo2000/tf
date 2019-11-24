## pip install google_images_download
## usage: python download_google_images.py object_name
import sys
## To fix SSL error of running url_open
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

object_name = sys.argv[1]

## Download Data 
from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()
arguments = {"keywords":object_name,"limit":250,"print_urls":False,"format":"jpg", "size":">400*300"}
#arguments = {"keywords":object_name,"limit":250,"print_urls":False,"format":"jpg", "size":">400*300", "chromedriver":r"C:\Program Files\Git\usr\bin\chromedriver.exe"}
paths = response.download(arguments)
