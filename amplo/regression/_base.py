#  Copyright (c) 2022 by Amplo.

from abc import ABCMeta

from amplo.base import BaseEstimator, LoggingMixin
from amplo.base.exceptions import NotFittedError

__all__ = ["BaseRegressor"]


class BaseRegressor(BaseEstimator, LoggingMixin, metaclass=ABCMeta):
    _estimator_type = "regressor"

    def __init__(self, model, verbose=0):
        BaseEstimator.__init__(self)
        LoggingMixin.__init__(self, verbose=verbose)

        self.model = model

    def predict(self, x, y=None, **predict_params):
        if not self.is_fitted_:
            raise NotFittedError
        return self.model.predict(x, **predict_params).reshape(-1)

    def fit_predict(self, x, y=None, **fit_params):
        self.model.fit(x, y)
        self.is_fitted_ = True
        return self.predict(x)

    def score(self, x, y):
        if not self.is_fitted_:
            raise NotFittedError
        return self.model.score(x, y)

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
