import pytest

from tests import make_data as _make_data
from tests import make_x_y as _make_x_y
from tests import rmfile, rmtree


@pytest.fixture(autouse=True)
def rmtree_automl():
    folder = "AutoML"
    rmtree(folder, must_exist=False)
    yield folder
    rmtree(folder, must_exist=False)


@pytest.fixture(autouse=True)
def rmfile_automl():
    file = "AutoML.log"
    yield file
    rmfile(file, must_exist=False)


@pytest.fixture
def make_x_y(mode):
    yield _make_x_y(mode=mode)


@pytest.fixture
def make_data(mode, make_x_y, target="target"):
    yield _make_data(mode=mode, target=target)
