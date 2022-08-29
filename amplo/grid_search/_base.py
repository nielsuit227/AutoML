#  Copyright (c) 2022 by Amplo.

import multiprocessing as mp
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _BaseScorer  # noqa
from sklearn.model_selection import KFold

from amplo.base.objects import LoggingMixin
from amplo.utils import check_dtypes

__all__ = ["BaseGridSearch"]


class BaseGridSearch(LoggingMixin):
    """
    Abstract base class for grid search.

    Purposes:
        - Enforces to inheriting classes to implement all abstract methods.
        - Defines the hyperparameter search space as it's the same for all
          grid search methods.

    Parameters
    ----------
    model : amplo.automl.modelling.ModelType
        Model object to optimize.
    params : optional
        Parameters to optimize. Has no effect for `OptunaGridSearch`.
    n_trials : int
        Limit the number of trials/candidates to search.
    timeout : int
        Limit the time for optimization.
    cv : sklearn.model_selection.BaseCrossValidator
        Cross validation object.
    scoring : str or sklearn.metrics._scorer._BaseScorer
        A valid string for `sklearn.metrics.get_scorer`.
    verbose : int
        Verbose logging.
    """

    def __init__(
        self,
        model,
        params=None,
        n_trials=250,
        timeout=None,
        cv=KFold(n_splits=10),
        scoring="accuracy",
        verbose=0,
    ):
        super().__init__(verbose=verbose)

        # Input tests
        if hasattr(model, "is_fitted") and model.is_fitted:
            warn(
                "The model is already fitted but Amplo's grid search will re-fit.",
                UserWarning,
            )
        scoring = get_scorer(scoring)
        check_dtypes("scoring", scoring, _BaseScorer)

        # Set class attributes
        self.model = model
        self.params = params
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.scoring = get_scorer(scoring)

        self.x, self.y = None, None
        self.binary = True
        self.samples = None

        # Model specific settings
        if type(self.model).__name__ == "LinearRegression":
            self.n_trials = 1

    @property
    def _hyper_parameter_values(
        self,
    ) -> Dict[str, Tuple[str, List[Union[str, float]], Optional[int]]]:
        """Get model specific hyper parameter values, indicating predefined
        search areas to optimize.

        Notes
        -----

        Each item of the output dictionary consists of:
            - parameter name (str), and
            - parameter specifications (tuple)
                - parameter type (str)
                - parameter arguments (list)
                - number of distinct values (int, optional) [only for exhaustive grid
                  search]

        Parameter types include:
            - 'categorical': categorical values
            - 'int': discretized uniform value space
            - 'logint': discretized logarithmic uniform value space
            - 'uniform': uniform value space
            - 'loguniform': logarithmic uniform value space

        Parameter arguments are:
            - [categorical]: a list of all options
            - [int, logint, uniform, loguniform]: a tuple with min and max value

        **Special case (conditionals):**
            In some cases, one wants to grid-search certain parameters only
            if another parameter condition is present. Such conditions are
            specified via the dedicated key 'CONDITIONALS'.
        """

        # Extract model name & type
        model_name = type(self.model).__name__
        model_type = re.split(r"Regressor|Classifier", model_name)[0]
        # Determine whether it's classification or regression
        is_regression = bool(re.match(r".*(Regression|Regressor|SVR)", model_name))
        is_classification = bool(
            re.match(r".*(Classification|Classifier|SVC)", model_name)
        )
        if not (is_regression or is_classification):
            raise ValueError("Could not determine mode (regression or classification).")

        # Define min-max-function
        def minimax(min_, value, max_):
            return max(min_, min(value, max_))

        # Find matching model and return its parameter values

        if model_name == "LinearRegression" or model_name == "LogisticRegression":
            return {}

        elif model_name == "Lasso" or "Ridge" in model_name:
            return dict(
                alpha=("uniform", [0, 10], 25),
            )

        elif model_name in ("SVR", "SVC"):
            return dict(
                gamma=("categorical", ["scale", "auto", 0.001, 0.01, 0.1, 0.5, 1]),
                C=("uniform", [0.001, 10], 25),
            )

        elif model_type == "KNeighbors":
            return dict(
                n_neighbors=("int", [5, minimax(5, self.samples // 10, 50)], 5),
                weights=("categorical", ["uniform", "distance"]),
                leaf_size=("int", [1, minimax(1, self.samples // 10, 100)], 5),
                n_jobs=("categorical", [mp.cpu_count() - 1]),
            )

        elif model_type == "MLP":
            raise NotImplementedError("MLP is not supported")

        elif model_type == "SGD":
            params = dict(
                loss=(
                    "categorical",
                    [
                        "squared_loss",
                        "huber",
                        "epsilon_insensitive",
                        "squared_epsilon_insensitive",
                    ],
                ),
                penalty=("categorical", ["l2", "l1", "elasticnet"]),
                alpha=("uniform", [0, 10], 5),
                max_iter=("int", [250, 1000], 3),
            )
            if is_classification:
                params.update(
                    loss=(
                        "categorical",
                        ["hinge", "log", "modified_huber", "squared_hinge"],
                    )
                )
            return params

        elif model_type == "DecisionTree":
            params = dict(
                criterion=(
                    "categorical",
                    ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                ),
                max_depth=("int", [3, minimax(3, int(np.log2(self.samples)), 25)], 4),
            )
            if is_classification:
                params.update(criterion=("categorical", ["gini", "entropy"]))
            return params

        elif model_type == "AdaBoost":
            params = dict(
                n_estimators=("int", [25, 250], 5),
                loss=("categorical", ["linear", "square", "exponential"]),
                learning_rate=("loguniform", [0.001, 1], 10),
            )
            if is_classification:
                params.pop("loss", None)
            return params

        elif model_type == "Bagging":
            params = dict(
                n_estimators=("int", [10, 250], 4),
                max_samples=("uniform", [0.5, 1.0], 4),
                max_features=("uniform", [0.5, 1.0], 4),
                n_jobs=("categorical", [mp.cpu_count() - 1]),
            )
            return params

        elif model_type == "CatBoost":
            params = dict(
                n_estimators=("int", [500, 2000], 5),
                verbose=("categorical", [0]),
                od_pval=("categorical", [1e-5]),
                loss_function=("categorical", ["MAE", "RMSE"]),
                learning_rate=("loguniform", [0.001, 0.5], 5),
                l2_leaf_reg=("uniform", [0, 10], 5),
                depth=("int", [3, minimax(3, int(np.log2(self.samples)), 10)], 4),
                min_data_in_leaf=("int", [1, minimax(1, self.samples // 10, 1000)], 5),
                grow_policy=(
                    "categorical",
                    ["SymmetricTree", "Depthwise", "Lossguide"],
                ),
            )
            if is_classification:
                params.update(
                    loss_function=(
                        "categorical",
                        ["Logloss" if self.binary else "MultiClass"],
                    )
                )
            return params

        elif model_type == "GradientBoosting":
            params = dict(
                loss=("categorical", ["ls", "lad", "huber"]),
                learning_rate=("loguniform", [0.001, 0.5], 10),
                max_depth=("int", [3, minimax(3, int(np.log2(self.samples)), 10)], 4),
                n_estimators=("int", [100, 1000], 4),
                min_samples_leaf=(
                    "int",
                    [1, minimax(1, int(self.samples / 10), 1000)],
                    3,
                ),
                max_features=("uniform", [0.5, 1], 3),
                subsample=("uniform", [0.5, 1], 3),
            )
            if is_classification:
                params.update(loss=("categorical", ["deviance", "exponential"]))
            return params

        elif model_type == "HistGradientBoosting":
            params = dict(
                loss=("categorical", ["least_squares", "least_absolute_deviation"]),
                learning_rate=("loguniform", [0.001, 0.5], 10),
                max_iter=("int", [100, 250], 4),
                max_leaf_nodes=("int", [30, 150], 4),
                max_depth=("int", [3, minimax(3, int(np.log2(self.samples)), 10)], 4),
                min_samples_leaf=(
                    "int",
                    [1, minimax(1, int(self.samples / 10), 1000)],
                    4,
                ),
                l2_regularization=("uniform", [0, 10], 5),
                max_bins=("int", [100, 255], 4),
                early_stopping=("categorical", [True]),
            )
            if is_classification:
                params.pop("loss", None)
            return params

        elif model_type == "RandomForest":
            params = dict(
                n_estimators=("int", [50, 1000], 5),
                criterion=("categorical", ["squared_error", "absolute_error"]),
                max_depth=("int", [3, minimax(3, int(np.log2(self.samples)), 15)], 4),
                max_features=("categorical", ["auto", "sqrt"]),
                min_samples_split=("int", [2, 50], 4),
                min_samples_leaf=("int", [1, minimax(1, self.samples // 10, 1000)], 5),
                bootstrap=("categorical", [True, False]),
            )
            if is_classification:
                params.update(criterion=("categorical", ["gini", "entropy"]))
            return params

        elif model_type == "XGB":
            params = dict(
                objective=("categorical", ["reg:squarederror"]),
                eval_metric=("categorical", ["rmse"]),
                booster=("categorical", ["gbtree", "gblinear", "dart"]),
                alpha=("loguniform", [1e-8, 1], 10),
                learning_rate=("loguniform", [0.001, 0.5], 5),
                n_jobs=("categorical", [mp.cpu_count() - 1]),
            )
            params["lambda"] = ("loguniform", [1e-8, 1], 10)
            if is_classification:
                params.update(
                    # objective=("categorical", ["multi:softprob"]),
                    eval_metric=("categorical", ["logloss"]),
                )
            params["CONDITIONALS"] = dict(
                booster=[
                    (
                        "gbtree",
                        dict(
                            max_depth=(
                                "int",
                                [1, minimax(1, int(np.log2(self.samples)), 10)],
                                5,
                            ),
                            eta=("loguniform", [1e-8, 1], 5),
                            gamma=("loguniform", [1e-8, 1], 5),
                            grow_policy=("categorical", ["depthwise", "lossguide"]),
                        ),
                    ),
                    (
                        "dart",
                        dict(
                            max_depth=(
                                "int",
                                [1, minimax(1, int(np.log2(self.samples)), 10)],
                                5,
                            ),
                            eta=("loguniform", [1e-8, 1], 5),
                            gamma=("loguniform", [1e-8, 1], 5),
                            grow_policy=("categorical", ["depthwise", "lossguide"]),
                            sample_type=("categorical", ["uniform", "weighted"]),
                            normalize_type=("categorical", ["tree", "forest"]),
                            rate_drop=("loguniform", [1e-8, 1], 5),
                            skip_drop=("loguniform", [1e-8, 1], 5),
                        ),
                    ),
                ],
            )
            return params

        elif model_type == "LGBM":
            if is_regression:
                return dict(
                    num_leaves=("int", [10, 150], 5),
                    min_data_in_leaf=(
                        "int",
                        [1, minimax(1, self.samples // 10, 1000)],
                        0,
                    ),
                    min_sum_hessian_in_leaf=("uniform", [0.001, 0.5], 0),
                    colsample_bytree=("uniform", [0, 1], 5),
                    reg_alpha=("uniform", [0, 1], 5),
                    reg_lambda=("uniform", [0, 1], 5),
                    verbosity=("categorical", [-1]),
                    n_jobs=("categorical", [mp.cpu_count() - 1]),
                )
            else:  # is_classification
                return dict(
                    objective=(
                        "categorical",
                        ["binary" if self.binary else "multiclass"],
                    ),
                    metric=(
                        "categorical",
                        ["binary_error", "auc", "average_precision", "binary_logloss"]
                        if self.binary
                        else ["multi_error", "multi_logloss", "auc_mu"],
                    ),
                    boosting_type=("categorical", ["gbdt"]),
                    lambda_l1=("loguniform", [1e-8, 10], 4),
                    lambda_l2=("loguniform", [1e-8, 10], 4),
                    num_leaves=("int", [10, 5000], 4),
                    max_depth=("int", [5, 20], 4),
                    min_data_in_leaf=(
                        "int",
                        [1, minimax(1, self.samples // 10, 1000)],
                        0,
                    ),
                    min_gain_to_split=("uniform", [0, 5], 0),
                    feature_fraction=("uniform", [0.4, 1], 0),
                    bagging_fraction=("uniform", [0.4, 1], 0),
                    bagging_freq=("int", [1, 7], 0),
                    verbosity=("categorical", [-1]),
                    n_jobs=("categorical", [mp.cpu_count() - 1]),
                )

        # Raise error if no match was found
        raise NotImplementedError(
            "Hyper parameter tuning not implemented for {}".format(model_name)
        )

    def get_parameter_min_max(self) -> pd.DataFrame:
        """
        Get all min and max values from model-specific set of parameters.

        Omit categorical parameters as min and max values are ambiguous.

        Returns
        -------
        param_min_max : pd.DataFrame
            Min and max values.
        """

        # Get all model's parameters
        param_values = self._hyper_parameter_values
        # Pop conditionals and integrate all into `param_values`
        conditionals = param_values.pop("CONDITIONALS", {})
        for check_p_name, check_p_criteria in conditionals.items():
            for matching_value, additional_params in check_p_criteria:
                for name, value in additional_params.items():
                    param_values[name] = value

        # Filter for min and max in non-categorical parameters
        param_min_max = {}
        for p_name, value in param_values.items():
            p_type, p_args = value[:2]
            if p_type != "categorical":  # ("int", "logint", "uniform", "loguniform")
                min_, max_ = p_args
                param_min_max[p_name] = {"min": min_, "max": max_}

        # Combine all values to pd.DataFrame
        param_min_max = pd.DataFrame(param_min_max).T
        return param_min_max

    @abstractmethod
    def _get_hyper_params(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Get grid search specific distributions or samples.

        This function translates `self._hyper_parameter_values` to the expected
        format for the given grid search.

        Parameters
        ----------
        args : optional
            Grid search specific arguments.
        kwargs : optional
            Grid search specific keyword arguments.

        Returns
        -------
        Grid search parameters
        """
        pass

    @abstractmethod
    def fit(self, x, y) -> pd.DataFrame:
        """
        Run fit with model-specific set of parameters.

        Parameters
        ----------
        x : array
            Data features.
        y : array
            Data labels

        Returns
        -------
        results : pd.DataFrame
            Results of the grid search.
        """
        pass
