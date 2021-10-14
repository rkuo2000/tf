import requests

url = 'http://127.0.0.1:5000/csv'
#url='https://b2bc-123-195-196-86.ngrok.io/csv'

f = open("0_000.csv", "rb")

r = requests.post(url, files = {"csv_file": ('0_000.csv', f, 'text/csv')})

print(r.text)
