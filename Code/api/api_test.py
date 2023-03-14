import base64
import requests

image_name = 'test.jpg'

with open(image_name, 'rb') as img:
    byte_string = base64.b64encode(img.read().decode('utf-8'))

res = requests.post('http://127.0.0.1:8080/test', json={'data': byte_string})
print(res.json())
