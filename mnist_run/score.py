import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from run import Classifier


def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_dir = Path(os.getenv('AZUREML_MODEL_DIR'))
    model_ckpt_filepath = list(model_dir.glob("**/*.ckpt"))[0]
    model = Classifier.load_from_checkpoint(model_ckpt_filepath.as_posix())
    model.eval()
    model.freeze()


def run(input_data):
    input_data = torch.tensor(json.loads(input_data)['data'])

    classes = [
        'zero',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine'
    ]

    output = model(input_data)
    pred_probs = F.softmax(output, dim=1)[0]
    index = torch.argmax(pred_probs, dim=0)

    result = {
        "label": classes[index].title(),
        "probability": str(pred_probs[index].item())
    }

    return result