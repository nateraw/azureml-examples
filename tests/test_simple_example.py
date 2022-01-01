import json
from pathlib import Path

import pytest
from simple_example import score, run


@pytest.mark.parametrize(
    "message,expected", [("Yo!!", "Yo!!"), (None, "Hello, world!")]
)
@pytest.mark.usefixtures("rm_logdir")
def test_local_run(message, expected):
    args = run.parse_args(f"--message {message}".split() if message is not None else "")
    run.main(args)
    assert Path("logs/message.txt").read_text() == expected

    score.init()
    request_data = json.dumps({"data": "blah"})
    response_data = score.run(request_data)
    assert response_data == {"message": expected, "input_data": "blah"}
