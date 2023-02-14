#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from amplo.base.exceptions import NotFittedError
from amplo.base.objects import BaseEstimator
from amplo.classification._base import BaseClassifier


class PartialBoostingClassifier(BaseClassifier):
    """
    Amplo wrapper for classification boosting models.

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
        "AdaBoostClassifier",
        "GradientBoostingClassifier",
        "LGBMClassifier",
        "XGBClassifier",
        "CatBoostClassifier",
    ]

    def __init__(self, model, step, verbose=0):
        model = deepcopy(model)
        super().__init__(model=model, verbose=verbose)

        model_name = type(model).__name__
        if model_name not in self._SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model {model_name}")
        if model_name in ("AdaBoostClassifier", "GradientBoostingClassifier"):
            model.estimators_ = model.estimators_[:step]

        self.step = int(step)
        self.classes_ = model.classes_

    def _get_prediction_kwargs(self):
        model_name = type(self.model).__name__
        if model_name in ("AdaBoostClassifier", "GradientBoostingClassifier"):
            return {}
        elif model_name == "LGBMClassifier":
            return {"num_iterations": self.step}
        elif model_name == "XGBClassifier":
            return {"iteration_range": (0, self.step)}
        elif model_name == "CatBoostClassifier":
            return {"ntree_end": self.step}
        else:
            raise AttributeError(f"Unsupported model {model_name}")

    def fit(self, x: pd.DataFrame, y: pd.Series, *args, **kwargs):
        self.classes_ = y.unique()
        return self.model.fit(x, y)

    def predict(self, x: pd.DataFrame, y: pd.Series | None = None, **kwargs):
        return self.model.predict(x, **kwargs, **self._get_prediction_kwargs())

    def predict_proba(self, x: pd.DataFrame, *args, **kwargs):
        if not self.is_fitted_:
            raise NotFittedError
        return self.model.predict_proba(x, **kwargs, **self._get_prediction_kwargs())

    @staticmethod
    def n_estimators(model: BaseEstimator) -> int:
        model_name = type(model).__name__
        if model_name in ("AdaBoostClassifier", "GradientBoostingClassifier"):
            return len(model.estimators_)  # type: ignore
        elif model_name in ("LGBMClassifier", "XGBClassifier"):
            return model.model.n_estimators
        elif model_name == "CatBoostClassifier":
            return model.model.tree_count_
        else:
            raise ValueError(f"Unsupported model {model_name}")
