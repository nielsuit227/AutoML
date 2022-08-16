#  Copyright (c) 2022 by Amplo.

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from amplo.automl import Modeller

__all__ = [
    "rmtree",
    "rmfile",
    "make_x_y",
    "make_data",
    "get_all_modeller_models",
    "find_first_warning_of_type",
    "RandomPredictor",
    "OverfitPredictor",
    "DelayedRandomPredictor",
]


def rmtree(folder="Auto_ML", must_exist=False):
    if Path(folder).exists():
        shutil.rmtree(folder)
    elif must_exist:
        raise FileNotFoundError(f"Directory {folder} does not exist")


def rmfile(file: str, must_exist=False):
    Path(file).unlink(missing_ok=not must_exist)


def make_x_y(mode: str):
    if mode == "classification":
        x, y = make_classification(n_features=5)
    elif mode == "multiclass":
        x, y = make_classification(n_features=5, n_classes=3, n_informative=3)
    elif mode == "regression":
        x, y = make_regression(n_features=5)
    else:
        raise ValueError("Invalid mode")
    x, y = pd.DataFrame(x), pd.Series(y)
    x.columns = [f"feature_{i}" for i in range(len(x.columns))]
    y.name = "target"
    return x, y


def make_data(mode: str, target="target"):
    data, y = make_x_y(mode)
    data[target] = y
    return data


def get_all_modeller_models(mode: str, **kwargs):
    models = {  # get each model type only once (!) with dictionary trick
        type(model).__name__: model
        for model in [
            *Modeller(mode=mode, samples=100, **kwargs).return_models(),
            *Modeller(mode=mode, samples=100_000, **kwargs).return_models(),
        ]
    }.values()
    return list(models)


def find_first_warning_of_type(typ, record):
    """
    Find first warning of type ``typ`` in warnings record.

    Parameters
    ----------
    typ : Any
        Warning to search for.
    record : pytest.WarningsRecorder
        Warnings record from ``with pytest.warns(...) as record`` context manager.

    Returns
    -------
    pytest.WarningsMessage
    """
    for item in record:
        if issubclass(item.category, typ):
            return item
    raise ValueError(f"No warning of type {typ} found in warnings record.")


# ----------------------------------------------------------------------
# Dummies


class _DummyPredictor:
    """
    Dummy predictor for testing.

    Parameters
    ----------
    mode : str
        Predicting mode ("classification" or "regression").
    """

    _dummy_classifier = None
    _dummy_regressor = None

    def __init__(self, mode):
        assert self._dummy_classifier is not None, "Dummy not set"
        assert self._dummy_regressor is not None, "Dummy not set"

        if mode == "classification":
            self.predictor = self._dummy_classifier()
        elif mode == "regression":
            self.predictor = self._dummy_regressor()
        else:
            raise ValueError("Invalid predictor mode.")

    def fit(self, x, y):
        return self.predictor.fit(x, y)

    def predict(self, x):
        return self.predictor.predict(x)

    def predict_proba(self, x):
        assert isinstance(self.predictor, self._dummy_classifier)
        return self.predictor.predict_proba(x)

    @property
    def classes_(self):
        if hasattr(self.predictor, "classes"):
            return self.predictor.classes


class _RandomClassifier:
    """
    Dummy classifier for testing.
    """

    def __init__(self):
        self.classes = None

    def fit(self, x, y):
        self.classes = np.unique(y)

    def predict(self, x):
        return np.random.choice(self.classes, len(x))

    def predict_proba(self, x):
        size = len(x), len(self.classes)
        proba = np.random.uniform(size=size)
        return proba * (1.0 / proba.sum(1)[:, np.newaxis])  # normalize


class _RandomRegressor:
    """
    Dummy regressor for testing.
    """

    def __init__(self):
        self.range = None

    def fit(self, x, y):
        self.range = np.min(y), np.max(y)

    def predict(self, x):
        return np.random.uniform(*self.range, len(x))


class RandomPredictor(_DummyPredictor):
    _dummy_classifier = _RandomClassifier
    _dummy_regressor = _RandomRegressor


class _OverfitClassifier:
    """
    Dummy classifier for testing. Returns the class if present in the data, else
    predicts 0
    """

    def __init__(self):
        self.classes = None
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x.to_numpy()
        self.y = y
        self.classes = y.unique()

    def predict(self, x):
        yt = []
        for i, row in x.iterrows():
            ind = np.where((row.values == self.x).all(axis=1))[0]
            if len(ind) == 0:
                yt.append(-1)
            else:
                yt.append(self.y.iloc[ind[0]])
        return yt

    def predict_proba(self, x):
        yt = []
        zeroes = [0 for _ in range(len(self.classes))]
        for i, row in x.iterrows():
            ind = np.where((row.values == self.x).all(axis=1))[0]
            if len(ind) == 0:
                yt.append(zeroes)
            else:
                yt.append(
                    [
                        0 if self.y.iloc[ind[0]] != i else 1
                        for i in range(len(self.classes))
                    ]
                )
        return yt


class _OverfitRegressor:
    """
    Dummy regressor for testing.
    """

    def __init__(self):
        self.classes = None
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x.to_numpy()
        self.y = y
        self.classes = y.unique()

    def predict(self, x):
        yt = []
        for i, row in x.iterrows():
            ind = np.where((row.values == self.x).all(axis=1))[0]
            if len(ind) == 0:
                yt.append(-1)
            else:
                yt.append(self.y.iloc[ind[0]])
        return yt


class OverfitPredictor(_DummyPredictor):
    _dummy_classifier = _OverfitClassifier
    _dummy_regressor = _OverfitRegressor


class DelayedRandomPredictor(RandomPredictor):
    def __init__(self, delay: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay

    def predict(self, x):
        time.sleep(self.delay)
        return super().predict(x)

    def predict_proba(self, x):
        time.sleep(self.delay)
        return super().predict_proba(x)
