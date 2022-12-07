#  Copyright (c) 2022 by Amplo.

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

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
@pytest.mark.parametrize("mode", ["classification", "regression"])
def make_x_y(request, mode):
    if mode == "classification":
        x, y = make_classification(n_features=5)
    elif mode == "multiclass":
        x, y = make_classification(n_features=5, n_classes=3, n_informative=3)
    elif mode == "regression":
        x, y = make_regression(n_features=5, noise=0.3)  # type: ignore
    else:
        raise ValueError("Invalid mode")
    x, y = pd.DataFrame(x), pd.Series(y)
    x.columns = [f"feature_{i}" for i in range(len(x.columns))]  # type: ignore
    y.name = "target"
    request.mode = mode
    yield x, y


@pytest.fixture
def make_data(request, make_x_y, target="target"):
    data, y = make_x_y
    data[target] = y
    request.data = data
    yield data


@pytest.fixture
def make_rng(request):
    request.cls.rng = np.random.default_rng(seed=92938)
    yield


@pytest.fixture(scope="class", params=["regression", "classification"])
def make_mode(request):
    mode = request.param
    target = "target"
    if mode == "classification":
        x, y = make_classification(n_features=5)
        request.cls.objective = "neg_log_loss"
    elif mode == "multiclass":
        x, y = make_classification(n_features=5, n_classes=3, n_informative=3)
        request.cls.objective = "neg_log_loss"
    elif mode == "regression":
        x, y = make_regression(n_features=5)  # type: ignore
        request.cls.objective = "neg_mean_squared_error"
    else:
        raise ValueError("Invalid mode")
    x, y = pd.DataFrame(x), pd.Series(y)
    x = x.rename({col: f"feature_{col}" for col in x.columns}, axis=1)
    request.cls.mode = mode
    request.cls.target = target
    request.cls.data = pd.concat([x, y.to_frame(target)], axis=1)
    yield
