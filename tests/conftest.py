#  Copyright (c) 2022 by Amplo.

import pytest

from tests import make_data as _make_data
from tests import make_x_y as _make_x_y
from tests import rmfile, rmtree


@pytest.fixture(autouse=True)
def rmtree_automl():
    folder = "Auto_ML"
    rmtree(folder, must_exist=False)
    yield folder
    rmtree(folder, must_exist=False)


@pytest.fixture(autouse=True)
def rmfile_automl():
    file = "AutoML.log"
    yield file
    try:
        rmfile(file, must_exist=False)
    except PermissionError:
        pass


@pytest.fixture
def make_x_y(mode):
    yield _make_x_y(mode=mode)


@pytest.fixture
def make_data(mode, make_x_y, target="target"):
    yield _make_data(mode=mode, target=target)
