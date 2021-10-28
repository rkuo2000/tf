import requests

url = 'http://127.0.0.1:5000/predict'
#url = 'http://192.168.1.5:5000/predict'
#url='https://b2bc-123-195-196-86.ngrok.io/csv'

f = open("0_000.csv", "rb")

r = requests.post(url, files = {"file": f})

print(r.text)
