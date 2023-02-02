#  Copyright (c) 2022 by Amplo.

from abc import ABCMeta
from typing import Any

import numpy.typing as npt
import pandas as pd

from amplo.base import BaseEstimator, LoggingMixin
from amplo.base.exceptions import NotFittedError

__all__ = ["BaseClassifier"]


class BaseClassifier(BaseEstimator, LoggingMixin, metaclass=ABCMeta):
    _estimator_type = "classifier"

    def __init__(self, model: BaseEstimator, verbose=0):
        BaseEstimator.__init__(self)
        LoggingMixin.__init__(self, verbose=verbose)

        self.model = model

    def predict(self, x: pd.DataFrame, **predict_params) -> npt.NDArray[Any]:
        if not self.is_fitted_:
            raise NotFittedError
        return self.model.predict(x, **predict_params).reshape(-1)

    def fit_predict(self, x: pd.DataFrame, y: pd.Series, *args, **fit_params):
        self.model.fit(x, y)
        return self.model.predict(x)

    def predict_proba(self, x: pd.DataFrame, *args, **kwargs) -> npt.NDArray[Any]:
        if not self.is_fitted_:
            raise NotFittedError
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model has no attribute `predict_proba`.")

        return self.model.predict_proba(x, *args, **kwargs)

    def _get_model_params(self, deep=True):
        """Gets JSON serializable model parameters only."""
        return self.model.get_params(deep=deep)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        model_params = self._get_model_params(deep=deep)

        return {**model_params, **params}

    def set_params(self, **params):
        # Set class params
        valid_class_params = super().get_params(deep=True)
        class_params = {k: v for k, v in params.items() if k in valid_class_params}
        super().set_params(**class_params)
        # Set model params
        model_params = {k: v for k, v in params.items() if k in valid_class_params}
        self.model.set_params(**model_params)
        return self
