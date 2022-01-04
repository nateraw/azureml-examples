import sys
from pathlib import Path
from unittest.mock import patch

import fire
import pytest

from keras_mnist_example import score, train


@pytest.mark.usefixtures('rm_logdir')
def test_local_run():
    args = [".py", "--max_epochs", "1"]
    with patch.object(sys, 'argv', args):
        fire.Fire(train.main)

    saved_model_path = Path('logs/saved_model')
    assert saved_model_path.exists()
    assert saved_model_path.is_dir()

    score.init()
    request_data = score.get_example_data()
    response_data = score.run(request_data)
    assert response_data['predictions'] == [6, 7]
