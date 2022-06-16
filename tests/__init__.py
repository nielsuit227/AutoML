import shutil
import os

import numpy as np


def rmtree(folder="AutoML", must_exist=False):
    if must_exist and not os.path.exists(folder):
        raise FileNotFoundError(f"Directory {folder} does not exist")
    if os.path.exists(folder):
        shutil.rmtree(folder)


# ----------------------------------------------------------------------
# Dummies


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


class RandomPredictor:
    """
    Dummy predictor for testing.

    Parameters
    ----------
    mode : str
        Predicting mode ("classification" or "regression").
    """

    def __init__(self, mode):
        if mode == "classification":
            self.predictor = _RandomClassifier()
        elif mode == "regression":
            self.predictor = _RandomRegressor()
        else:
            raise ValueError("Invalid predictor mode.")

    def fit(self, x, y):
        return self.predictor.fit(x, y)

    def predict(self, x):
        return self.predictor.predict(x)

    def predict_proba(self, x):
        assert isinstance(self.predictor, _RandomClassifier)
        return self.predictor.predict_proba(x)

    @property
    def classes_(self):
        if hasattr(self.predictor, "classes"):
            return self.predictor.classes


class OverfitPredictor:
    """
    Dummy predictor for testing.

    Parameters
    ----------
    mode : str
        Predicting mode ("classification" or "regression").
    """

    def __init__(self, mode):
        if mode == "classification":
            self.predictor = _OverfitClassifier()
        elif mode == "regression":
            self.predictor = _OverfitRegressor()
        else:
            raise ValueError("Invalid predictor mode.")

    def fit(self, x, y):
        return self.predictor.fit(x, y)

    def predict(self, x):
        return self.predictor.predict(x)

    def predict_proba(self, x):
        assert isinstance(self.predictor, _OverfitClassifier)
        return self.predictor.predict_proba(x)

    @property
    def classes_(self):
        if hasattr(self.predictor, "classes"):
            return self.predictor.classes


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
