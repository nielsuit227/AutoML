import shutil
import os

import numpy as np


def rmtree(folder='AutoML', must_exist=False):
    if must_exist and not os.path.exists(folder):
        raise FileNotFoundError(f'Directory {folder} does not exist')
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
