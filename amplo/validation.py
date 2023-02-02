#  Copyright (c) 2022 by Amplo.
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator, cross_val_predict

from amplo.base import BaseEstimator, LoggingMixin

__all__ = ["ModelValidator"]


class ModelValidator(LoggingMixin):
    """
    Model validator.

    Parameters
    ----------
    cv_splits : int, default: 5
        Number of cross validation splits.
    cv_shuffle : bool, default: True
        Whether to shuffle the data for cross validation.
    verbose : {0, 1, 2}, default: 1
        Verbosity for logging.
    """

    def __init__(self, target: str, cv: BaseCrossValidator, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.target = target
        self.cv = cv

    def validate(self, model: BaseEstimator, data: pd.DataFrame, mode: str):
        """
        Validate model and return performance metrics.

        Parameters
        ----------
        model :
            Model to be validated.
        data : pd.DataFrame
            Training data for validation.
        mode : {"classification", "regression"}
            Model mode.

        Returns
        -------
        metrics : dict
            Cross validated model performance metrics.
        """
        model = deepcopy(model)
        y = data[self.target]
        x = data.drop(self.target, axis=1)

        # Calculate metrics
        if mode == "classification":
            metrics = self._validate_classification(model, x, y)
        elif mode == "regression":
            metrics = self._validate_regression(model, x, y)
        else:
            raise NotImplementedError("Invalid mode for validation.")

        # Logging
        for name, value in metrics.items():
            # We expect a mean and std value
            if not isinstance(value, float):
                continue
            name = f"{name.replace('_', ' ').title()}:".ljust(20)
            self.logger.info(f"{name} {value:.5f}")

        if mode == "classification":
            self.logger.info(
                """Confusion Matrix:
            Predicted / actual  |    Positive    |    Negative    |
            Positive            |  {}  |  {}  |
            Negative            |  {}  |  {}  |
            """.format(
                    f"{metrics.get('true_positives')}".rjust(12),
                    f"{metrics.get('false_positives')}".rjust(12),
                    f"{metrics.get('false_negatives')}".rjust(12),
                    f"{metrics.get('true_negatives')}".rjust(12),
                )
            )

        return metrics

    def _validate_classification(self, model, x, y):
        """
        Cross validate classification (binary or multiclass) model.

        Parameters
        ----------
        model :
            Model to be validated.
        x : np.ndarray
            Training data for validation.
        y : np.ndarray
            Target data for validation.

        Returns
        -------
        metrics : dict
            Cross validated performance metrics of model.
        """
        labels = np.unique(y)
        log_loss_ = None

        # Modelling
        if hasattr(model, "predict_proba"):
            yp = cross_val_predict(model, x, y, cv=self.cv, method="predict_proba")
            log_loss_ = log_loss(y, yp)
            assert isinstance(yp, np.ndarray) and yp.shape == (len(y), len(labels))
            yp = np.argmax(yp, axis=1)
        else:
            yp = cross_val_predict(model, x, y, cv=self.cv)

        # Summarize metrics
        tp = np.sum(np.logical_and(y == 1, yp == 1))
        fp = np.sum(np.logical_and(y == 0, yp == 1))
        tn = np.sum(np.logical_and(y == 0, yp == 0))
        fn = np.sum(np.logical_and(y == 1, yp == 0))
        metrics = {
            "log_loss": log_loss_,
            "roc_auc": roc_auc_score(y, yp, labels=labels),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "accuracy": accuracy_score(y, yp),
            "precision": tp / (tp + fp),
            "sensitivity_": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "f1_score": f1_score(y, yp, labels=labels, average="weighted"),
        }

        return metrics

    def _validate_regression(self, model, x, y):
        """
        Cross validate regression model.

        Parameters
        ----------
        model :
            Model to be validated.
        x : np.ndarray
            Training data for validation.
        y : np.ndarray
            Target data for validation.

        Returns
        -------
        metrics : dict
            Cross validated performance metrics of model.
        """
        # Modelling
        yp = cross_val_predict(model, x, y)

        # Summarize metrics
        metrics = {
            "mean_relative_error": mean_absolute_percentage_error(y, yp),
            "r2_score": r2_score(y, yp),
            "mean_absolute_error": mean_absolute_error(y, yp),
            "mean_squared_error": mean_squared_error(y, yp),
            "max_error": max_error(y, yp),
        }

        return metrics
