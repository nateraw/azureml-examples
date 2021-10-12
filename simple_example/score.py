import json
import os

import fire
import requests
from pathlib import Path


def init():
    global message

    model_dir = Path(os.getenv("AZUREML_MODEL_DIR", "logs/"))
    message_filepath = model_dir / "message.txt"
    message = message_filepath.read_text()


def run(input_data):
    input_data = json.loads(input_data)["data"]
    return {"message": message, "input_data": input_data}


def main(endpoint: str = None):
    request_data = json.dumps({"data": "blah"}) # if args.request_data is None else args.request_data

    if endpoint is not None:
        response_data = requests.post(endpoint, request_data, headers={"Content-Type": "application/json"}).json()
    else:
        init()
        response_data = run(request_data)

    print(response_data)
    return response_data


if __name__ == "__main__":
    fire.Fire(main)
