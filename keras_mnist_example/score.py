import json
import os
import requests
from argparse import ArgumentParser
from base64 import b64decode, b64encode
from pathlib import Path
from PIL import Image

import numpy as np
from tensorflow import keras


def init():
    global model
    model_dir = Path(os.getenv('AZUREML_MODEL_DIR', 'logs/'))
    model = keras.models.load_model(str(model_dir / 'saved_model'))


def run(input_data):
    images = json.loads(input_data).get('images')

    if isinstance(images, str):
        images = [images]
    
    x = np.stack([bytestr_to_image(x) for x in images]).astype('float32') / 255

    out = model.predict(x)
    preds = out.argmax(1)

    return {'predictions': preds.tolist(), 'proba': out.tolist()}


def image_to_bytestr(x):
    return b64encode(x.tobytes()).decode("utf-8")


def bytestr_to_image(x):
    decoded = b64decode(x.encode("utf-8"))
    return Image.frombytes("L", (28, 28), decoded)


def get_example_data():
    img_filepaths = ['images/6.png', 'images/7.png']
    image_bytes = []
    for img_filepath in img_filepaths:
        im = Image.open(img_filepath)
        image_bytes.append(image_to_bytestr(im))
    return json.dumps({'images': image_bytes})


def main():
    parser = ArgumentParser()
    parser.add_argument('--endpoint', type=str, default=None)
    args = parser.parse_args()

    request_data = get_example_data()

    if args.endpoint is not None:
        response_data = requests.post(args.endpoint, request_data, headers={'Content-Type': 'application/json'}).json()
    else:
        init()
        response_data = run(request_data)
    
    return response_data


if __name__ == '__main__':
    response_data = main()
