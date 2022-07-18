#  Copyright (c) 2022 by Amplo.

from copy import deepcopy
from warnings import warn

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from amplo.base import LoggingMixin

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

    def __init__(self, cv_splits=5, cv_shuffle=True, verbose=1):
        super().__init__(verbose=verbose)
        self.cv_splits = cv_splits
        self.cv_shuffle = cv_shuffle

    def validate(self, model, x, y, mode):
        """
        Validate model and return performance metrics.

        Parameters
        ----------
        model :
            Model to be validated.
        x : array_like
            Training data for validation.
        y : array_like
            Target data for validation.
        mode : {"classification", "regression"}
            Model mode.

        Returns
        -------
        metrics : dict
            Cross validated model performance metrics.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # Calculate metrics
        if mode == "classification":
            metrics = self._validate_classification(model, x, y)
        elif mode == "regression":
            metrics = self._validate_regression(model, x, y)
        else:
            warn("Invalid mode for validation.", UserWarning)
            metrics = {}

        # Logging
        for name, values in metrics.items():
            # We expect a mean and std value
            if not isinstance(values, list) or np.array(values).size != 2:
                continue

            name = f"{name.replace('_', ' ').title()}:".ljust(20)
            self.logger.info(f"{name} {values[0]:.2f} \u00B1 {values[1]:.2f}")

        cm = np.array(metrics.get("confusion_matrix", []))
        if mode == "classification" and cm.shape == (2, 2, 2):
            cm_text = []
            for mean, std in zip(cm[0].ravel(), cm[1].ravel()):
                cm_text.append(f"{mean:.2f} \u00B1 {std:.2f}".ljust(12))

            for line in [
                "Confusion Matrix:",
                "  actual / predicted  |    Negative    |    Positive    |",
                f"            Negative  |  {cm_text[0]}  |  {cm_text[1]}  |",
                f"            Positive  |  {cm_text[2]}  |  {cm_text[3]}  |",
            ]:
                self.logger.info(line)

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
        is_binary = labels.size == 2

        # Initialize metrics
        accuracy = []
        precision = []
        f1 = []
        cm = []
        log_loss_ = []

        # Binary metrics
        roc_auc = []
        sensitivity = []
        specificity = []

        # Modelling
        cv = StratifiedKFold(
            n_splits=self.cv_splits, shuffle=self.cv_shuffle, random_state=38234
        )
        for t, v in cv.split(x, y):
            xt, yt = x[t], y[t].reshape(-1)
            xv, yv = x[v], y[v].reshape(-1)
            cv_model = deepcopy(model)
            cv_model.fit(xt, yt)
            yv_pred = cv_model.predict(xv).reshape(-1)

            # Metrics
            accuracy.append(accuracy_score(yv, yv_pred))
            cm.append(confusion_matrix(yv, yv_pred, labels=labels))
            if hasattr(cv_model, "predict_proba"):
                log_loss_.append(log_loss(yv, cv_model.predict_proba(xv)))

            if is_binary:
                roc_auc.append(roc_auc_score(yv, yv_pred, labels=labels))
                tn, fp, fn, tp = cm[-1].ravel()
                if tp + fp > 0:
                    precision.append(precision_score(yv, yv_pred, labels=labels))
                if tp + fn > 0:
                    sensitivity.append(tp / (tp + fn))
                if tn + fp > 0:
                    specificity.append(tn / (tn + fp))
                if tp + fp > 0 and tp + fn > 0:
                    f1.append(f1_score(yv, yv_pred, labels=labels))
            else:
                precision.append(
                    precision_score(yv, yv_pred, labels=labels, average="weighted")
                )
                f1.append(f1_score(yv, yv_pred, labels=labels, average="weighted"))

        # Summarize metrics
        metrics = {}
        for name, values in [
            ("accuracy", accuracy),
            ("precision", precision),
            ("sensitivity", sensitivity),
            ("specificity", specificity),
            ("f1_score", f1),
            ("log_loss", log_loss_),
        ]:
            if len(values) == 0:
                # Skip empty values (for multi-class)
                continue
            metrics[name] = [np.mean(values), np.std(values)]

        # Add confusion matrix metrics
        cm_totals = np.sum(cm, axis=(1, 2), keepdims=True)
        cm_means = np.mean(cm / cm_totals, axis=0)
        cm_stds = np.std(cm / cm_totals, axis=0)
        del cm_totals
        metrics["confusion_matrix"] = [cm_means.tolist(), cm_stds.tolist()]

        if is_binary:
            metrics["true_positives"] = [cm_means[0, 0], cm_stds[0, 0]]
            metrics["false_positives"] = [cm_means[0, 1], cm_stds[0, 1]]
            metrics["true_negatives"] = [cm_means[1, 0], cm_stds[1, 0]]
            metrics["false_negatives"] = [cm_means[1, 1], cm_stds[1, 1]]

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

        # Initialize metrics
        mae = []
        mse = []
        r2 = []
        max_error_ = []
        mean_rel_error = []

        # Modelling
        cv = KFold(n_splits=self.cv_splits, shuffle=self.cv_shuffle, random_state=29283)
        for t, v in cv.split(x, y):
            xt, yt = x[t], y[t].reshape(-1)
            xv, yv = x[v], y[v].reshape(-1)
            cv_model = deepcopy(model)
            cv_model.fit(xt, yt)
            yv_pred = cv_model.predict(xv).reshape(-1)

            # Metrics
            mae.append(mean_absolute_error(yv, yv_pred))
            mse.append(mean_squared_error(yv, yv_pred))
            r2.append(r2_score(yv, yv_pred))
            max_error_.append(max_error(yv, yv_pred))
            mean_rel_error.append(mean_absolute_percentage_error(yv, yv_pred))

        # Summarize metrics
        metrics = {}
        for name, values in [
            ("mean_absolute_error", mae),
            ("mean_squared_error", mse),
            ("r2_score", r2),
            ("max_error", max_error_),
            ("mean_relative_error", mean_rel_error),
        ]:
            metrics[name] = [np.mean(values), np.std(values)]

        return metrics
