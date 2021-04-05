import json
import os
from pathlib import Path


def init():
    global message
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_dir = Path(os.getenv('AZUREML_MODEL_DIR'))
    message_filepath = model_dir / 'message.txt'
    message = message_filepath.read_text()


def run(input_data):
    input_data = json.loads(input_data)['data']
    return {'message': message, 'input_data': input_data}
