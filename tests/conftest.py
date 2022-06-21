import pytest

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

from tests import rmtree


@pytest.fixture(autouse=True)
def rmtree_automl():
    folder = 'AutoML'
    rmtree(folder, must_exist=False)
    yield folder
    rmtree(folder, must_exist=False)


@pytest.fixture
def make_x_y(mode):
    if mode == 'classification':
        x, y = make_classification()
    elif mode == 'regression':
        x, y = make_regression()
    else:
        raise ValueError('Invalid mode')
    yield pd.DataFrame(x), pd.Series(y)
