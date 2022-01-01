import json
import os
from argparse import ArgumentParser
from base64 import b64decode, b64encode
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

try:
    from model import Classifier
except ImportError:
    from .model import Classifier


def init():
    global model, transform
    transform = ToTensor()
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR", "logs/"))
    model_ckpt_filepath = list(model_dir.glob("**/*.ckpt"))[0]
    model = Classifier.load_from_checkpoint(str(model_ckpt_filepath))
    model.freeze()


def image_to_bytestr(x):
    return b64encode(x.tobytes()).decode("utf-8")


def bytestr_to_image(x):
    decoded = b64decode(x.encode("utf-8"))
    return Image.frombytes("L", (28, 28), decoded)


def run(input_data):
    data = json.loads(input_data)
    images = data.get("images")

    if images is None:
        return "KeyError: no key named 'images' in given request body"

    if isinstance(images, str):
        images = [images]

    x = torch.stack([transform(bytestr_to_image(x)) for x in images])
    out = model(x)
    proba = F.softmax(out, dim=1)
    pred = torch.argmax(proba, dim=1)
    return {"predictions": pred.tolist(), "proba": proba.tolist()}


def get_example_data():
    img_filepaths = ["images/6.png", "images/7.png"]
    image_bytes = []
    for img_filepath in img_filepaths:
        im = Image.open(img_filepath)
        image_bytes.append(image_to_bytestr(im))
    return json.dumps({"images": image_bytes})


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--endpoint", type=str, default=None)
    return parser.parse_args(args)


def main(args):

    request_data = get_example_data()

    if args.endpoint is not None:
        response_data = requests.post(
            args.endpoint, request_data, headers={"Content-Type": "application/json"}
        ).json()
    else:
        init()
        response_data = run(request_data)

    print(f"{'Predicted':10s}", response_data["predictions"])

    return response_data


if __name__ == "__main__":
    args = parse_args()
    response_data = main(args)
