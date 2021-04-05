import requests
import json


data = json.dumps({'data': 'blah'})
endpoint = '<YOUR SIMPLE DEPLOY ENDPOINT HERE>'
headers = {'Content-Type': 'application/json'}
response= requests.post(endpoint, data, headers=headers)
print("Response", response.text)

# Response:
# '{"message": "Howdy!", "input_data": "blah"}'