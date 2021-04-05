import json
import os
import requests

from torchvision.datasets import MNIST
from torchvision import transforms


if __name__ == '__main__':
    endpoint = "<YOUR ENDPOINT HERE>"

    dataset = MNIST(
        os.getcwd(),
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    x, y = dataset[0]
    x = x.flatten(1).tolist()

    data = json.dumps({'data': x})
    headers = {'Content-Type': 'application/json'}

    response = requests.post(endpoint, data, headers=headers)
    print('Response:', response.text)
    print('-' * 65)
    print('Actual Label:', y)