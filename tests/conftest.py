#  Copyright (c) 2022 by Amplo.

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from tests import rmfile, rmtree


@pytest.fixture(autouse=True)
def rmtree_automl() -> Iterator[str]:
    folder = "Auto_ML"
    rmtree(folder, must_exist=False)
    yield folder
    rmtree(folder, must_exist=False)


@pytest.fixture(autouse=True)
def rmfile_automl() -> Iterator[str]:
    file_ = "AutoML.log"
    yield file_
    try:
        rmfile(file_, must_exist=False)
    except PermissionError:
        pass


@pytest.fixture
def x_y(request, mode: str) -> Iterator[tuple[pd.DataFrame, pd.Series]]:
    if mode == "classification":
        x, y = make_classification(n_features=5)
    elif mode == "multiclass":
        x, y = make_classification(n_features=5, n_classes=3, n_informative=3)
    elif mode == "regression":
        x, y = make_regression(n_features=5, noise=0.3)
    else:
        raise ValueError("Invalid mode")
    x, y = pd.DataFrame(x), pd.Series(y)
    x.columns = [f"feature_{i}" for i in range(len(x.columns))]
    y.name = "target"
    request.mode = mode
    yield x, y


@pytest.fixture
def data(
    request, x_y: tuple[pd.DataFrame, pd.Series], target="target"
) -> Iterator[pd.DataFrame]:
    data, y = x_y
    data[target] = y
    request.data = data
    yield data


@pytest.fixture
def classification_data() -> Iterator[pd.DataFrame]:
    x, y = make_classification(n_features=5, flip_y=0)
    df = pd.DataFrame(x, columns=["a", "b", "c", "d", "e"])
    df["target"] = y
    yield df


@pytest.fixture
def multiindex_data(classification_data: pd.DataFrame) -> Iterator[pd.DataFrame]:
    log_ind = range(len(classification_data) // 10)
    yield classification_data.sort_values(by="target").set_index(
        pd.MultiIndex.from_product([range(10), log_ind]), drop=True
    )


@pytest.fixture
def freq_data() -> Iterator[pd.DataFrame]:
    pos = np.real(np.fft.ifft(np.array([0, 1, 0.1, 0.001, -1, -0.1, -0.001]), n=100))
    neg = np.real(np.fft.ifft(np.array([0, 0.1, 0.2, 0.1, -1, -0.1, -0.001]), n=100))
    yield pd.DataFrame(
        {
            "a": np.hstack([pos] * 10 + [neg] * 10),
            "target": np.hstack((np.zeros(1000), np.ones(1000))),
        },
        index=pd.MultiIndex.from_product([range(20), range(100)]),
    )


@pytest.fixture
def random_number_generator(request) -> Iterator[None]:
    request.cls.rng = np.random.default_rng(seed=92938)
    yield
