#  Copyright (c) 2022 by Amplo.
from __future__ import annotations

import os
import time
from copy import deepcopy
from datetime import datetime
from typing import Optional, TypeVar

import joblib
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model, metrics, model_selection, svm

from amplo.classification import CatBoostClassifier, LGBMClassifier, XGBClassifier
from amplo.regression import CatBoostRegressor, LGBMRegressor, XGBRegressor
from amplo.utils import check_dtypes
from amplo.utils.logging import get_root_logger

__all__ = ["ClassificationType", "Modeller", "ModelType", "RegressionType"]


logger = get_root_logger().getChild("Modeller")


ClassificationType = TypeVar(
    "ClassificationType",
    CatBoostClassifier,
    ensemble.BaggingClassifier,
    linear_model.RidgeClassifier,
    linear_model.LogisticRegression,
    LGBMClassifier,
    svm.SVC,
    XGBClassifier,
)
RegressionType = TypeVar(
    "RegressionType",
    CatBoostRegressor,
    ensemble.BaggingRegressor,
    linear_model.LinearRegression,
    LGBMRegressor,
    svm.SVR,
    XGBRegressor,
)
ModelType = TypeVar(
    "ModelType",
    CatBoostClassifier,
    CatBoostRegressor,
    ensemble.BaggingClassifier,
    ensemble.BaggingRegressor,
    linear_model.LinearRegression,
    linear_model.LogisticRegression,
    linear_model.RidgeClassifier,
    LGBMClassifier,
    LGBMRegressor,
    svm.SVC,
    svm.SVR,
    XGBClassifier,
    XGBRegressor,
)


class Modeller:
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
    mode : str
        Model mode. Either `regression` or `classification`.
    shuffle : bool
        Whether to shuffle samples from training / validation.
    n_splits : int
        Number of cross-validation splits.
    objective : str
        Performance metric to optimize. Must be a valid string for
        `sklearn.metrics.get_scorer`.
    samples : int
        Hypothetical number of samples in dataset. Useful to manipulate behavior
        of `return_models()` function.
    needs_proba : bool
        Whether the modelling needs a probability.
    folder : str
        Folder to store models and/or results.
    dataset : str
        Name of feature set. For documentation purposes only.
    store_models : bool
        Whether to store the trained models. If true, `folder` must be
        specified to take effect.
    store_results : bool
        Whether to store the results. If true, `folder` must be specified.

    See Also
    --------
    [Sklearn scorers](https://scikit-learn.org/stable/modules/model_evaluation.html
    """

    def __init__(
        self,
        mode: str = "regression",
        cv: Optional[model_selection.BaseCrossValidator] = None,
        objective: str = "accuracy",
        samples: Optional[int] = None,
        needs_proba: bool = True,
        folder: str = "",
        dataset: str = "set_0",
        store_models: bool = False,
        store_results: bool = True,
    ):
        if mode not in ("classification", "regression"):
            raise ValueError(f"Unsupported mode: {mode}")

        # Parameters
        self.cv = cv
        self.objective = objective
        self.mode = mode
        self.samples = samples
        self.dataset = str(dataset)
        self.store_results = store_results
        self.store_models = store_models
        self.needs_proba = needs_proba
        self.results = pd.DataFrame(
            columns=[
                "date",
                "model",
                "dataset",
                "params",
                "score",
                "worst_case",
                "time",
            ]
        )

        # Update CV if not provided
        if self.cv is None:
            if self.mode == "classification":
                self.cv = model_selection.StratifiedKFold(n_splits=3)
            elif self.mode == "regression":
                self.cv = model_selection.KFold(n_splits=3)

        # Folder
        self.folder = folder if len(folder) == 0 or folder[-1] == "/" else folder + "/"
        if (store_results or store_models) and self.folder != "":
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

    def fit(self, x, y):
        # Copy number of samples
        self.samples = len(y)

        # Convert to NumPy
        x = np.array(x)
        y = np.array(y).ravel()

        if self.store_results and "Initial_Models.csv" in os.listdir(self.folder):
            self.results = pd.read_csv(self.folder + "Initial_Models.csv")
            for i in range(len(self.results)):
                self.print_results(self.results.iloc[i])

        else:

            # Models
            self.models = self.return_models()

            # Loop through models
            for model in self.models:

                # Time & loops through Cross-Validation
                t_start = time.time()
                scores = model_selection.cross_val_score(
                    model, x, y, scoring=self.objective
                )
                score = sum(scores) / len(scores)
                run_time = time.time() - t_start

                # Append results
                result = {
                    "date": datetime.today().strftime("%d %b %y"),
                    "model": type(model).__name__,
                    "dataset": self.dataset,
                    "params": model.get_params(),
                    "score": score,
                    "worst_case": np.mean(scores) - np.std(scores),
                    "time": run_time,
                }
                self.results = pd.concat(
                    [self.results, pd.Series(result).to_frame().T], ignore_index=True
                )
                self.print_results(result)

                # Store model
                if self.store_models:
                    joblib.dump(
                        model,
                        self.folder
                        + type(model).__name__
                        + "_{:.4f}.joblib".format(score),
                    )

            # Store CSV
            if self.store_results:
                self.results.to_csv(self.folder + "Initial_Models.csv")

        # Return results
        return self.results

    def return_models(self):
        """
        Get all models that are considered appropriate for training.

        Returns
        -------
        list of ModelType
            Models that apply for given dataset size and mode.
        """
        models = []

        # All classifiers
        if self.mode == "classification":
            # The thorough ones
            if self.samples < 25000:
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
            if self.samples < 25000:
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

    def print_results(self, result):
        logger.info(
            "{} {}: {}    training time: {:.1f} s".format(
                result["model"].ljust(30),
                self.objective,
                f"{result['worst_case']:.4f}".ljust(15),
                result["time"],
            )
        )
