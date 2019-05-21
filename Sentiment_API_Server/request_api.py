import requests
from requests import get
url = 'http://127.0.0.1:5000/api'
params = dict(
    texts="This is a test")

# r = requests.get(url=url,json=params)
r = requests.post(url,json=params)

print(r.json())
