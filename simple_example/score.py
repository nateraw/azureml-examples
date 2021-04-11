import json
import os
from pathlib import Path


def init():
    global message

    model_dir = Path(os.getenv('AZUREML_MODEL_DIR'))
    message_filepath = model_dir / 'message.txt'
    message = message_filepath.read_text()


def run(input_data):
    input_data = json.loads(input_data)['data']
    return {'message': message, 'input_data': input_data}
