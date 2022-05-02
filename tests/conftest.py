import pytest

from tests import rmtree


@pytest.fixture(autouse=True)
def rmtree_automl():
    folder = 'AutoML'
    rmtree(folder, must_exist=False)
    yield folder
    rmtree(folder, must_exist=False)
