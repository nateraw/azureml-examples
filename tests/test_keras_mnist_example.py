from pathlib import Path

import pytest
from keras_mnist_example import score, train


@pytest.mark.usefixtures("rm_logdir")
def test_local_run():
    args = train.parse_args("--max_epochs 1".split())
    train.main(args)

    saved_model_path = Path("logs/saved_model")
    assert saved_model_path.exists()
    assert saved_model_path.is_dir()

    score.init()
    request_data = score.get_example_data()
    response_data = score.run(request_data)
    assert response_data["predictions"] == [6, 7]
