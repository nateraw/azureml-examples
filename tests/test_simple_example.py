import json
import sys
from pathlib import Path
from unittest.mock import patch

import fire
import pytest

from simple_example import score, run


@pytest.mark.parametrize(
    "message,expected", [(None, 'Hello, world!'), ('Yo!!', 'Yo!!')]
)
@pytest.mark.usefixtures('rm_logdir')
def test_local_run(message, expected):
    args = [".py"]
    if message:
        args += [message]
    with patch.object(sys, 'argv', args):
        fire.Fire(run.main)

    assert Path('logs/message.txt').read_text() == expected

    score.init()
    request_data = json.dumps({'data': 'blah'})
    response_data = score.run(request_data)
    assert response_data == {'message': expected, 'input_data': 'blah'}
