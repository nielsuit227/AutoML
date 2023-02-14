#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model, model_selection, svm

from amplo import classification, regression
from amplo.base import BaseEstimator, LoggingMixin
from amplo.base.objects import Result
from amplo.classification import CatBoostClassifier, LGBMClassifier, XGBClassifier
from amplo.regression import CatBoostRegressor, LGBMRegressor, XGBRegressor
from amplo.utils.logging import get_root_logger

__all__ = ["Modeller", "get_model"]


logger = get_root_logger().getChild("Modeller")


def get_model(model_str: str) -> BaseEstimator:
    """Returns a model object given a model string"""
    model: BaseEstimator

    if "RandomForest" in model_str or "Bagging" in model_str:
        model = getattr(ensemble, model_str)()
    elif model_str == "SVC":
        model = svm.SVC(probability=True)
    elif model_str == "SVR":
        model = svm.SVR()
    elif "Logistic" in model_str or "Linear" in model_str or "Ridge" in model_str:
        model = getattr(linear_model, model_str)()
    elif "Classifier" in model_str:
        model = getattr(classification, model_str)()
    elif "Regressor" in model_str:
        model = getattr(regression, model_str)()
    else:
        raise ValueError("Model not recognized.")
    return model


class Modeller(LoggingMixin):
    """
    Modeller for classification or regression tasks.

    Supported models:
        - linear models from ``scikit-learn``
        - random forest from ``scikit-learn``
        - bagging model from ``scikit-learn``
        - ~~gradient boosting from ``scikit-learn``~~
        - ~~histogram-based gradient boosting from ``scikit-learn``~~
        - XGBoost from DMLC
        - Catboost from Yandex
        - LightGBM from Microsoft

    Parameters
    ----------
    target : str, optional
    mode : str
        Model mode. Either `regression` or `classification`.
    cv : sklearn.model_selection.BaseCrossValidator, optional
    objective : str
        Performance metric to optimize. Must be a valid string for
        `sklearn.metrics.get_scorer`.
    samples : int, optional
        Hypothetical number of samples in dataset. Useful to manipulate behavior
        of `return_models()` function.
    needs_proba : bool, default = True
        Whether the modelling needs a probability.
    feature_set : str, optional
        Used to label returned results
    model : str, optional
        Used to limit search
    verbose : int

    See Also
    --------
    [Sklearn scorers](https://scikit-learn.org/stable/modules/model_evaluation.html
    """

    def __init__(
        self,
        target: str | None = None,
        mode: str = "classification",
        cv: model_selection.BaseCrossValidator | None = None,
        objective: str | None = None,
        samples: int | None = None,
        needs_proba: bool = True,
        feature_set: str | None = None,
        model: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        if mode not in ("classification", "regression"):
            raise ValueError(f"Unsupported mode: {mode}")

        # Parameters
        self.target = target
        self.cv = cv
        self.objective = objective
        self.mode = mode
        self.samples = samples
        self.needs_proba = needs_proba
        self.results: list[Result] = []
        self.feature_set = feature_set if feature_set is not None else ""
        self.model = model

        # Update CV if not provided
        if self.cv is None:
            if self.mode == "classification":
                self.cv = model_selection.StratifiedKFold(n_splits=3)
            elif self.mode == "regression":
                self.cv = model_selection.KFold(n_splits=3)

    def fit(self, data: pd.DataFrame) -> list[Result]:
        if not self.target:
            raise ValueError("Can only fit when target is provided.")
        if self.target not in data:
            raise ValueError(f"Target column not in dataframe: {self.target}")

        self.samples = len(data)
        y = data[self.target]
        x = data.drop(self.target, axis=1)

        # Convert to NumPy
        x = np.array(x)
        y = np.array(y).ravel()

        # Models
        self.models = self.return_models()

        # Loop through models
        for model in self.models:

            # Time & loops through Cross-Validation
            t_start = time.time()
            scores = model_selection.cross_val_score(
                model, x, y, scoring=self.objective, cv=self.cv
            )
            score = sum(scores) / len(scores)
            run_time = time.time() - t_start

            # Append results
            result = Result(
                date=datetime.today().strftime("%d %b %y"),
                model=type(model).__name__,
                params=model.get_params(),
                feature_set=self.feature_set,
                score=score,
                worst_case=np.mean(scores) - np.std(scores),
                time=run_time,
            )
            self.results.append(result)
            self.logger.info(
                f"{result.model.ljust(30)} {self.objective}: "
                f"{result.worst_case:15.4f}    training time:"
                f" {result.time:.1f} s"
            )

        # Return results
        return self.results

    def return_models(self) -> list[BaseEstimator]:
        """
        Get all models that are considered appropriate for training.

        Returns
        -------
        list of ModelType
            Models that apply for given dataset size and mode.
        """
        if self.model:
            return [get_model(self.model)]

        models: list[BaseEstimator] = []

        # All classifiers
        if self.mode == "classification":
            # The thorough ones
            if not self.samples or self.samples < 25000:
                models.append(svm.SVC(kernel="rbf", probability=self.needs_proba))
                models.append(ensemble.BaggingClassifier())
                # models.append(ensemble.GradientBoostingClassifier()) == XG Boost
                models.append(XGBClassifier())

            # The efficient ones
            else:
                # models.append(ensemble.HistGradientBoostingClassifier()) == LGBM
                models.append(LGBMClassifier())

            # And the multifaceted ones
            if not self.needs_proba:
                models.append(linear_model.RidgeClassifier())
            else:
                models.append(linear_model.LogisticRegression())
            models.append(CatBoostClassifier())
            models.append(ensemble.RandomForestClassifier())

        elif self.mode == "regression":
            # The thorough ones
            if not self.samples or self.samples < 25000:
                models.append(svm.SVR(kernel="rbf"))
                models.append(ensemble.BaggingRegressor())
                # models.append(ensemble.GradientBoostingRegressor()) == XG Boost
                models.append(XGBRegressor())

            # The efficient ones
            else:
                # models.append(ensemble.HistGradientBoostingRegressor()) == LGBM
                models.append(LGBMRegressor())

            # And the multifaceted ones
            models.append(linear_model.LinearRegression())
            models.append(CatBoostRegressor())
            models.append(ensemble.RandomForestRegressor())

        return models
