#  Copyright (c) 2022 by Amplo.

from abc import ABCMeta

from amplo.base import BasePredictor, LoggingMixin

__all__ = ["BaseRegressor"]


class BaseRegressor(BasePredictor, LoggingMixin, metaclass=ABCMeta):
    _estimator_type = "regressor"

    def __init__(self, model, verbose=0):
        BasePredictor.__init__(self)
        LoggingMixin.__init__(self, verbose=verbose)

        self.model = model

    def _fit(self, x, y=None, **fit_params):
        self.model.fit(x, y)

    def _predict(self, x, y=None, **predict_params):
        return self.model.predict(x, **predict_params).reshape(-1)

    def score(self, x, y):
        return self.model.score(x, y)

    def _get_model_params(self, deep=True):
        """Gets JSON serializable model parameters only."""
        return self.model.get_params(deep=deep)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        model_params = self._get_model_params(deep=deep)

        return {**params, **model_params}

    def set_params(self, **params):
        # Set class params
        valid_class_params = super().get_params(deep=True)
        class_params = {key: params[key] for key in params if key in valid_class_params}
        super().set_params(**class_params)
        # Set model params
        model_params = {key: params[key] for key in params if key not in class_params}
        self.model.set_params(**model_params)
        return self
