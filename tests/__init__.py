#  Copyright (c) 2022 by Amplo.
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from amplo.automl import Modeller
from amplo.base import BaseEstimator

__all__ = [
    "rmtree",
    "rmfile",
    "get_all_modeller_models",
    "find_first_warning_of_type",
    "create_test_folders",
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


def create_data_frames(n_samples, n_features):
    dim = (int(n_samples / 2), n_features)
    columns = [f"Feature_{i}" for i in range(n_features)]
    df1 = pd.DataFrame(
        columns=columns,
        data=np.vstack((np.random.normal(0, 1, dim), np.random.normal(100, 1, dim))),
    )
    df2 = pd.DataFrame(
        columns=columns,
        data=np.vstack((np.random.normal(0, 1, dim), np.random.normal(-100, 1, dim))),
    )
    return df1, df2


def create_test_folders(directory: Path | str, n_samples, n_features):
    directory = Path(directory)
    # Make directories
    for sub_folder in ("Class_1", "Class_2"):
        (directory / sub_folder).mkdir(exist_ok=True, parents=True)

    # Create and save dataframes
    for i in range(140):
        df1, df2 = create_data_frames(n_samples, n_features)
        df1.to_csv(directory / f"Class_1/Log_{i}.csv", index=False)
        df2.to_csv(directory / f"Class_2/Log_{i}.csv", index=False)


# ----------------------------------------------------------------------
# Dummies


class _DummyPredictor(BaseEstimator):
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
        return self.predictor.predict_proba(x)

    @property
    def classes_(self):
        if hasattr(self.predictor, "classes"):
            return self.predictor.classes


class _RandomClassifier(BaseEstimator):
    """
    Dummy classifier for testing.
    """

    def __init__(self):
        self.classes_ = None

    def fit(self, x, y):
        self.classes_ = np.unique(y)

    def predict(self, x):
        assert self.classes_ is not None
        return np.random.choice(self.classes_, len(x))

    def predict_proba(self, x):
        assert self.classes_ is not None
        size = len(x), len(self.classes_)
        proba = np.random.uniform(size=size)
        return proba * (1.0 / proba.sum(1)[:, np.newaxis])  # normalize


class _RandomRegressor(BaseEstimator):
    """
    Dummy regressor for testing.
    """

    def __init__(self):
        self.range = None

    def fit(self, x, y):
        self.range = np.min(y), np.max(y)

    def predict(self, x):
        assert self.range
        return np.random.uniform(self.range, len(x))


class RandomPredictor(_DummyPredictor):
    _dummy_classifier = _RandomClassifier
    _dummy_regressor = _RandomRegressor


class _OverfitClassifier(BaseEstimator):
    """
    Dummy classifier for testing. Returns the class if present in the data, else
    predicts 0
    """

    def __init__(self):
        self.classes_: np.ndarray | None = None
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x.to_numpy()
        self.y = y
        self.classes_ = y.unique()

    def predict(self, x):
        assert self.y is not None
        yt = []
        for i, row in x.iterrows():
            ind = np.where((row.values == self.x).all(axis=1))[0]
            if len(ind) == 0:
                yt.append(-1)
            else:
                yt.append(self.y.iloc[ind[0]])
        return np.array(yt)

    def predict_proba(self, x):
        assert self.classes_ is not None and self.y is not None
        yt = []
        zeroes = [0 for _ in range(len(self.classes_))]
        for i, row in x.iterrows():
            ind = np.where((row.values == self.x).all(axis=1))[0]
            if len(ind) == 0:
                yt.append(zeroes)
            else:
                yt.append(
                    [
                        0 if self.y.iloc[ind[0]] != i else 1
                        for i in range(len(self.classes_))
                    ]
                )
        return np.array(yt)


class _OverfitRegressor(_DummyPredictor):
    """
    Dummy regressor for testing.
    """

    def __init__(self):
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x.to_numpy()
        self.y = y

    def predict(self, x):
        assert self.y is not None and self.x is not None
        yt = []
        for i, row in x.iterrows():
            ind = np.where((row.values == self.x).all(axis=1))[0]
            if len(ind) > 0:
                yt.append(ind[0])
            else:
                yt.append(-1000)
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
