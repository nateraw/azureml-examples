from pathlib import Path
from shutil import rmtree

import pytest


@pytest.fixture()
def rm_logdir():
    if Path("logs/").exists():
        rmtree("logs/")
    print("Removing Logdir")
