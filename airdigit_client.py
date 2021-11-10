import requests

url = 'http://127.0.0.1:5000/predict'
#url = 'http://192.168.1.7:5000/predict'
#url='https://b2bc-123-195-196-86.ngrok.io/predict'

f = open("0_000.csv", "rb")

r = requests.post(url, files = {"file": f})

print(r.text)
