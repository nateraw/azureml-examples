import json
import os
from argparse import ArgumentParser
from pathlib import Path

import requests


def init():
    global message

    model_dir = Path(os.getenv("AZUREML_MODEL_DIR", "logs/"))
    message_filepath = model_dir / "message.txt"
    message = message_filepath.read_text()


def run(input_data):
    input_data = json.loads(input_data)["data"]
    return {"message": message, "input_data": input_data}


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--endpoint", type=str, default=None)
    return parser.parse_args(args)


def main(args):
    request_data = json.dumps({"data": "blah"})  # if args.request_data is None else args.request_data

    if args.endpoint is not None:
        response_data = requests.post(args.endpoint, request_data, headers={"Content-Type": "application/json"}).json()
    else:
        init()
        response_data = run(request_data)

    print(response_data)
    return response_data


if __name__ == "__main__":
    main(parse_args())
