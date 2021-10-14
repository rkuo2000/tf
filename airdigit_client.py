import requests

url = 'http://192.168.1.6:5000/csv'
#url='https://b2bc-123-195-196-86.ngrok.io/csv'

f = open("0_000.csv", "rb")

r = requests.post(url, files = {"file": f})

print(r.text)
