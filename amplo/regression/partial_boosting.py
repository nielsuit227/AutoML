#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from amplo.regression._base import BaseRegressor


class PartialBoostingRegressor(BaseRegressor):
    """
    Amplo wrapper for regressor boosting models.

    The number of estimators being used in the prediction are limited.

    Parameters
    ----------
    model
        Boosting model to wrap.
    step : int
        Number of iterations/estimators to limit the model on predictions.
    verbose : {0, 1, 2}
        Verbose logging.
    """

    _SUPPORTED_MODELS = [
        "AdaBoostRegressor",
        "GradientBoostingRegressor",
        "LGBMRegressor",
        "XGBRegressor",
        "CatBoostRegressor",
    ]

    def __init__(self, model, step, verbose=0):
        model = deepcopy(model)

        model_name = type(model).__name__
        if model_name not in self._SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model {model_name}")
        if model_name in ("AdaBoostRegressor", "GradientBoostingRegressor"):
            model.estimators_ = model.estimators_[:step]

        super().__init__(model=model, verbose=verbose)
        self.step = int(step)

    def _get_prediction_kwargs(self):
        model_name = type(self.model).__name__
        if model_name in ("AdaBoostRegressor", "GradientBoostingRegressor"):
            return {}
        elif model_name == "LGBMRegressor":
            return {"num_iterations": self.step}
        elif model_name == "XGBRegressor":
            return {"iteration_range": (0, self.step)}
        elif model_name == "CatBoostRegressor":
            return {"ntree_end": self.step}
        else:
            raise AttributeError(f"Unsupported model {model_name}")

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        return self.model.fit(x, y, **kwargs)

    def predict(self, x: pd.DataFrame, y: pd.Series | None = None, **kwargs):
        return self.model.predict(x, **kwargs, **self._get_prediction_kwargs())

    @staticmethod
    def n_estimators(model):
        model_name = type(model).__name__
        if model_name in ("AdaBoostRegressor", "GradientBoostingRegressor"):
            return len(model.estimators_)
        elif model_name in ("LGBMRegressor", "XGBRegressor"):
            return model.model.n_estimators
        elif model_name == "CatBoostRegressor":
            return model.model.tree_count_
        else:
            raise ValueError(f"Unsupported model {model_name}")
