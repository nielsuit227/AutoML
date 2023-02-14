#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import copy
import re
import time
from datetime import datetime
from warnings import warn

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, cross_val_score

from amplo.base.objects import BaseEstimator, LoggingMixin, Result

__all__ = ["OptunaGridSearch"]


MIN_REGULARIZATION = 1e-6
MAX_REGULARIZATION = 1e2
MIN_BOOSTERS = 100
MAX_BOOSTERS = 2500
MIN_DEPTH = 1
MAX_DEPTH = 10
MIN_SPLIT = 0.1
MAX_SPLIT = 1.0
MIN_LEAF_SIZE = 1
MAX_LEAF_SIZE = 10_000


# Define min-max-function
def minimax(min_, value, max_):
    return max(min_, min(value, max_))


def max_depth(samples: int) -> int:
    return minimax(MIN_DEPTH, int(np.log2(samples)), MAX_DEPTH)


def max_leaf_size(samples: int) -> int:
    return minimax(MIN_LEAF_SIZE, int(samples / 10), MAX_LEAF_SIZE)


def warn_at_extreme(params: dict[str, int | float | str], samples: int):
    for param, value in params.items():
        if param in ["depth", "max_depth"]:
            if value == MIN_DEPTH or value == max_depth(samples):
                warn(
                    f"Parameter {param} (={value}) at the edge of search space: ({MIN_DEPTH}, {max_depth(samples)})"
                )
        if param in [
            "n_estimators",
            "alpha",
            "C",
            "l2_leaf_reg",
            "learning_rate",
            "eta",
            "gamma",
            "l2_regularization",
            "skip_drop",
            "rate_drop",
            "min_sum_hessian_in_leaf",
            "reg_alpha",
            "reg_lambda",
            "lambda_l1",
            "lambda_l1",
        ]:
            if value in (MIN_REGULARIZATION, MAX_REGULARIZATION):
                warn(
                    f"Parameter {param} (={value}) at the edge of search space: ({MIN_REGULARIZATION}, {MAX_REGULARIZATION})"
                )
        if param in ["max_iter", "n_estimators"]:
            if value in (MIN_BOOSTERS, MAX_BOOSTERS):
                warn(
                    f"Parameter {param} (={value}) at the edge of search space: ({MIN_BOOSTERS}, {MAX_BOOSTERS})"
                )
        if param in [
            "max_samples",
            "max_features",
            "colsample_bytree",
            "feature_fraction",
            "bagging_fraction",
            "subsample",
        ]:
            if value in (MIN_SPLIT, MAX_SPLIT):
                warn(
                    f"Parameter {param} (={value}) at the edge of search space: ({MIN_SPLIT}, {MAX_SPLIT})"
                )
        if param in ["leaf_size", "min_data_in_leaf", "min_samples_leaf"]:
            if value in (MIN_LEAF_SIZE, max_leaf_size(samples)):
                warn(
                    f"Parameter {param} (={value}) at the edge of search space: ({MIN_LEAF_SIZE}, {max_leaf_size(samples)})"
                )


class OptunaGridSearch(LoggingMixin):
    """
    Wrapper for ``optuna`` grid search.

    Takes any model supported by `Amplo.AutoML.Modelling` whose parameter
    search space is predefined for each model.
    Optimal choice [1]

    [1] https://arxiv.org/pdf/2201.06433.pdf

    Parameters
    ----------
    model : Amplo.AutoML.Modeller.ModelType
        Model object to optimize.
    n_trials : int
        Limit the number of trials/candidates to search.
    timeout : int
        Limit the time for optimization.
    cv : sklearn.model_selection.BaseCrossValidator
        Cross validation object.
    scoring : str or sklearn.metrics._scorer._BaseScorer, default = neg_log_loss
        A valid string for `sklearn.metrics.get_scorer`.
    verbose : int
        Verbose logging.
    """

    def __init__(
        self,
        model: BaseEstimator,
        target: str,
        n_trials: int = 250,
        timeout: int = -1,
        feature_set: str = "",
        cv: BaseCrossValidator = StratifiedKFold(n_splits=10),
        scoring: str = "neg_log_loss",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        # Input tests
        if hasattr(model, "is_fitted") and model.is_fitted:
            warn(
                "The model is already fitted but Amplo's grid search will re-fit.",
                UserWarning,
            )
        scoring = get_scorer(scoring)

        # Set class attributes
        self.target = target
        self.model = model
        self.n_trials = n_trials
        self.timeout = timeout
        self.feature_set = feature_set
        self.cv = cv
        self.scoring = get_scorer(scoring)

        # Set attributes
        self.binary_: bool | None = None
        self.samples_: int | None = None
        self.trial_count_ = -1

        # Model specific settings
        if type(self.model).__name__ in ("LinearRegression", "LogisticRegression"):
            self.n_trials = 1

    def _get_hyper_params(
        self, trial: optuna.Trial
    ) -> dict[str, None | bool | int | float | str]:
        """Get model specific hyper parameter values, indicating predefined
        search areas to optimize.

        parameters
        ----------
        trial : optuna.Trial
        """
        assert self.samples_ and self.binary_ and self.x
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

        # Find matching model and return its parameter values
        if model_name in ("LinearRegression", "LogisticRegression"):
            return {}

        elif model_name in ("Lasso", "Ridge"):
            return dict(
                alpha=trial.suggest_float(
                    "alpha", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                )
            )

        elif model_name in ("SVR", "SVC"):
            return dict(
                gamma=trial.suggest_categorical(
                    "gamma", ["scale", "auto", 0.001, 0.01, 0.1, 0.5, 1]
                ),
                C=trial.suggest_float(
                    "C", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                ),
            )

        elif model_type == "KNeighbors":
            return dict(
                n_neighbors=trial.suggest_int(
                    "n_neighbors", 5, minimax(5, self.samples_ // 10, 50)
                ),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                leaf_size=trial.suggest_int(
                    "leaf_size", MIN_LEAF_SIZE, max_leaf_size(self.samples_), log=True
                ),
            )

        elif model_type == "MLP":
            raise NotImplementedError("MLP is not supported")

        elif model_type == "SGD":
            params = dict(
                loss=trial.suggest_categorical(
                    "loss",
                    [
                        "squared_loss",
                        "huber",
                        "epsilon_insensitive",
                        "squared_epsilon_insensitive",
                    ],
                ),
                penalty=trial.suggest_categorical(
                    "penalty", ["l2", "l1", "elasticnet"]
                ),
                alpha=trial.suggest_float(
                    "alpha", MIN_REGULARIZATION, MAX_REGULARIZATION
                ),
                max_iter=trial.suggest_int(
                    "max_iter", MIN_BOOSTERS, MAX_BOOSTERS, log=True
                ),
            )
            if is_classification:
                params.update(
                    loss=trial.suggest_categorical(
                        "loss",
                        ["hinge", "log", "modified_huber", "squared_hinge"],
                    )
                )
            return params

        elif model_type == "DecisionTree":
            params = dict(
                criterion=trial.suggest_categorical(
                    "criterion",
                    ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                ),
                max_depth=trial.suggest_int(
                    "max_depth", MIN_DEPTH, max_depth(self.samples_)
                ),
            )
            if is_classification:
                params.update(
                    criterion=trial.suggest_categorical(
                        "criterion", ["gini", "entropy"]
                    )
                )
            return params

        elif model_type == "AdaBoost":
            params = dict(
                n_estimators=trial.suggest_int(
                    "n_estimators", MIN_BOOSTERS, MAX_BOOSTERS, log=True
                ),
                loss=trial.suggest_categorical(
                    "loss", ["linear", "square", "exponential"]
                ),
                learning_rate=trial.suggest_float(
                    "learning_rate", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                ),
            )
            if is_classification:
                params.pop("loss", None)
            return params

        elif model_type == "Bagging":
            return dict(
                n_estimators=trial.suggest_int(
                    "n_estimators", MIN_BOOSTERS, MAX_BOOSTERS, log=True
                ),
                max_samples=trial.suggest_float("max_samples", MIN_SPLIT, MAX_SPLIT),
                max_features=trial.suggest_float("max_features", MIN_SPLIT, MAX_SPLIT),
            )

        elif model_type == "CatBoost":
            params = dict(
                n_estimators=trial.suggest_int(
                    "n_estimators", MIN_BOOSTERS, MAX_BOOSTERS, log=True
                ),
                loss_function=trial.suggest_categorical(
                    "loss_function", ["MAE", "RMSE"]
                ),
                learning_rate=trial.suggest_float(
                    "learning_rate", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                ),
                l2_leaf_reg=trial.suggest_float(
                    "l2_leaf_reg", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                ),
                depth=trial.suggest_int("depth", MIN_DEPTH, max_depth(self.samples_)),
                min_data_in_leaf=trial.suggest_int(
                    "min_data_in_leaf",
                    MIN_LEAF_SIZE,
                    max_leaf_size(self.samples_),
                    log=True,
                ),
                grow_policy=trial.suggest_categorical(
                    "grow_policy",
                    ["SymmetricTree", "Depthwise", "Lossguide"],
                ),
                od_pval=1e-5,
                verbose=0,
            )
            if is_classification:
                params.update(
                    loss_function=trial.suggest_categorical(
                        "loss_function",
                        ["Logloss" if self.binary else "MultiClass"],
                    )
                )
            return params

        elif model_type == "GradientBoosting":
            params = dict(
                loss=trial.suggest_categorical("loss", ["ls", "lad", "huber"]),
                learning_rate=trial.suggest_float(
                    "learning_rate", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                ),
                max_depth=trial.suggest_int(
                    "max_depth", MIN_DEPTH, max_depth(self.samples_)
                ),
                n_estimators=trial.suggest_int(
                    "n_estimators", MIN_BOOSTERS, MAX_BOOSTERS, log=True
                ),
                min_samples_leaf=trial.suggest_int(
                    "min_samples_leaf",
                    MIN_LEAF_SIZE,
                    max_leaf_size(self.samples_),
                    log=True,
                ),
                max_features=trial.suggest_float(
                    "max_features", MIN_SPLIT, MAX_SPLIT, log=True
                ),
                subsample=trial.suggest_float(
                    "subsample", MIN_SPLIT, MAX_SPLIT, log=True
                ),
            )
            if is_classification:
                params.update(
                    loss=trial.suggest_categorical(
                        "categorical", ["deviance", "exponential"]
                    )
                )
            return params

        elif model_type == "HistGradientBoosting":
            params = dict(
                loss=trial.suggest_categorical(
                    "loss", ["least_squares", "least_absolute_deviation"]
                ),
                learning_rate=trial.suggest_float(
                    "learning_rate", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                ),
                max_iter=trial.suggest_int(
                    "max_iter", MIN_BOOSTERS, MAX_BOOSTERS, log=True
                ),
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 30, 150, log=True),
                max_depth=trial.suggest_int(
                    "max_depth", MIN_DEPTH, max_depth(self.samples_)
                ),
                min_samples_leaf=trial.suggest_int(
                    "min_samples_leaf",
                    MIN_LEAF_SIZE,
                    max_leaf_size(self.samples_),
                    log=True,
                ),
                l2_regularization=trial.suggest_float(
                    "l2_regularization",
                    MIN_REGULARIZATION,
                    MAX_REGULARIZATION,
                    log=True,
                ),
                max_bins=trial.suggest_int("max_bins", 100, 255),
                early_stopping=True,
            )
            if is_classification:
                params.pop("loss")
            return params

        elif model_type == "RandomForest":
            params = dict(
                n_estimators=trial.suggest_int(
                    "n_estimators", MIN_BOOSTERS, MAX_BOOSTERS
                ),
                criterion=trial.suggest_categorical(
                    "criterion", ["squared_error", "absolute_error"]
                ),
                max_depth=trial.suggest_int(
                    "max_depth", MIN_DEPTH, max_depth(self.samples_)
                ),
                max_features=trial.suggest_categorical(
                    "max_features", ["log2", "sqrt"]
                ),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 50),
                min_samples_leaf=trial.suggest_int(
                    "min_samples_leaf",
                    MIN_LEAF_SIZE,
                    max_leaf_size(self.samples_),
                    log=True,
                ),
                bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            )
            if is_classification:
                params.update(
                    criterion=trial.suggest_categorical(
                        "criterion", ["gini", "entropy"]
                    )
                )
            return params

        elif model_type == "XGB":
            params = dict(
                objective="reg:squarederror",
                eval_metric="rmse",
                booster=trial.suggest_categorical(
                    "booster", ["gbtree", "gblinear", "dart"]
                ),
                alpha=trial.suggest_float(
                    "alpha", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                ),
                learning_rate=trial.suggest_float(
                    "learning_rate", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                ),
            )
            if is_classification:
                params.update(
                    eval_metric="logloss",
                )
            if params.get("booster") == "gbtree":
                params.update(
                    max_depth=trial.suggest_int(
                        "max_depth",
                        MIN_DEPTH,
                        max_depth(self.samples_),
                    ),
                    eta=trial.suggest_float(
                        "eta", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                    ),
                    gamma=trial.suggest_float(
                        "gamma", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                    ),
                    grow_policy=trial.suggest_categorical(
                        "grow_policy", ["depthwise", "lossguide"]
                    ),
                )
            elif params.get("booster") == "dart":
                params.update(
                    max_depth=trial.suggest_int(
                        "max_depth",
                        MIN_DEPTH,
                        max_depth(self.samples_),
                    ),
                    eta=trial.suggest_float(
                        "eta", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                    ),
                    gamma=trial.suggest_float(
                        "gamma", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                    ),
                    grow_policy=trial.suggest_categorical(
                        "grow_policy", ["depthwise", "lossguide"]
                    ),
                    sample_type=trial.suggest_categorical(
                        "sample_type", ["uniform", "weighted"]
                    ),
                    normalize_type=trial.suggest_categorical(
                        "normalize_type", ["tree", "forest"]
                    ),
                    rate_drop=trial.suggest_float(
                        "rate_drop", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                    ),
                    skip_drop=trial.suggest_float(
                        "skip_drop", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                    ),
                )
            return params

        elif model_type == "LGBM":
            if is_regression:
                return dict(
                    num_leaves=trial.suggest_int("num_leaves", 10, 150),
                    min_data_in_leaf=trial.suggest_int(
                        "min_data_in_leaf",
                        MIN_LEAF_SIZE,
                        max_leaf_size(self.samples_),
                        log=True,
                    ),
                    min_sum_hessian_in_leaf=trial.suggest_float(
                        "min_sum_hessian_in_leaf",
                        MIN_REGULARIZATION,
                        MAX_REGULARIZATION,
                        log=True,
                    ),
                    colsample_bytree=trial.suggest_float(
                        "colsample_bytree", MIN_SPLIT, MAX_SPLIT, log=True
                    ),
                    reg_alpha=trial.suggest_float(
                        "reg_alpha", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                    ),
                    reg_lambda=trial.suggest_float(
                        "reg_lambda", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                    ),
                    verbosity=-1,
                )
            else:  # is_classification
                return dict(
                    objective="binary" if self.binary else "multiclass",
                    metric=trial.suggest_categorical(
                        "metric",
                        ["binary_error", "auc", "average_precision", "binary_logloss"]
                        if self.binary
                        else ["multi_error", "multi_logloss", "auc_mu"],
                    ),
                    boosting_type="gbdt",
                    lambda_l1=trial.suggest_float(
                        "lambda_l1", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                    ),
                    lambda_l2=trial.suggest_float(
                        "lambda_l2", MIN_REGULARIZATION, MAX_REGULARIZATION, log=True
                    ),
                    num_leaves=trial.suggest_int("num_leaves", 10, 5000),
                    max_depth=trial.suggest_int(
                        "max_depth", MIN_DEPTH, max_depth(self.samples_)
                    ),
                    min_data_in_leaf=trial.suggest_int(
                        "min_data_in_leaf",
                        MIN_LEAF_SIZE,
                        max_leaf_size(self.samples_),
                        log=True,
                    ),
                    min_gain_to_split=trial.suggest_float("min_gain_to_split", 0, 5),
                    feature_fraction=trial.suggest_float(
                        "feature_fraction", MIN_SPLIT, MAX_SPLIT, log=True
                    ),
                    bagging_fraction=trial.suggest_float(
                        "bagging_fraction", MIN_SPLIT, MAX_SPLIT, log=True
                    ),
                    bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
                    verbosity=-1,
                )

        # Raise error if no match was found
        raise NotImplementedError(
            "Hyper parameter tuning not implemented for {}".format(model_name)
        )

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        assert self.target in data
        self.y = data[self.target]
        self.x = data.drop(self.target, axis=1)

        # Set mode
        self.binary = self.y.nunique() == 2
        self.samples_ = len(self.y)

        # Set up study
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=236868),
            direction="maximize",
            pruner=_BadTrialPruner(2.0, 15),
        )
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[
                _StopStudyWhenConsecutivePruning(10),
                _StopStudyAfterNPruning(50),
            ],
        )

        # Parse results
        results: list[Result] = []
        date = datetime.today().strftime("%d %b %y")
        results = [
            Result(
                date=date,
                model=type(self.model).__name__,
                params=trial.params,
                score=trial.value if trial.value else -np.inf,
                worst_case=trial.value - trial.user_attrs["score_std"],
                feature_set=self.feature_set,
                time=trial.user_attrs["time"],
            )
            for trial in study.get_trials()
        ]
        self.trial_count_ = len(study.trials)
        results.sort(reverse=True)

        # Warn against edge params
        warn_at_extreme(results[0].params, self.samples_)

        return results

    def _objective(self, trial: optuna.Trial) -> float:
        # Make a copy
        model_copy = copy.deepcopy(self.model)

        # Cross validation
        t_start = time.time()
        scores = cross_val_score(model_copy, self.x, self.y, scoring=self.scoring)
        score = sum(scores) / len(scores)
        run_time = time.time() - t_start

        # Set manual metrics
        trial.set_user_attr("time", run_time)
        trial.set_user_attr("score_std", np.std(scores))

        # Stop study (avoid overwriting)
        if trial.number == self.n_trials:
            self.logger.info("Maximum trails reached.")
            trial.study.stop()
        if score > -1e-9:
            self.logger.info(f"Superb score achieved, stopping search ({score:.4E})")
            trial.study.stop()

        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned

        return score


# ----------------------------------------------------------------------
# Pruning


class _BadTrialPruner(optuna.pruners.BasePruner):
    """
    Pruner to detect outlying metrics of the trials.

    Prune if the mean intermediate value is worse than the best trial value minus (or
    plus) a multiple of its intermediate value standard deviation.

    Parameters
    ----------
    std_threshold_multiplier : float
        Multiplier for the best trials intermediate value std to define the pruning
        threshold.
    n_startup_trials : int
        Pruning is disabled until the given number of trials finish in the same study.
    """

    def __init__(self, std_threshold_multiplier, n_startup_trials):
        if n_startup_trials < 0:
            raise ValueError(
                f"Number of startup trials cannot be negative but got "
                f"{n_startup_trials}."
            )

        self._std_threshold_multiplier = std_threshold_multiplier
        self._n_startup_trials = n_startup_trials

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        # Don't prune while startup trials
        if not [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]:
            return False
        if len(study.best_trial.intermediate_values) < self._n_startup_trials:
            return False

        # Define pruning thresholds
        best_trial_value = study.best_trial.value
        best_trial_value_std = np.std(
            list(study.best_trial.intermediate_values.values())
        )
        threshold = (
            best_trial_value - self._std_threshold_multiplier * best_trial_value_std
        )

        # Pruning
        curr_trial_mean = np.mean(list(trial.intermediate_values.values()))
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            return curr_trial_mean < threshold
        else:
            raise RuntimeError(
                "Pruning with a threshold is arbitrary when the study direction is "
                "undefined."
            )


class _StopStudyWhenConsecutivePruning:
    """
    Optuna study callback that stops the study when trials keep being pruned.

    Parameters
    ----------
    threshold : int
        Critical threshold for consecutively pruned trials. Stops trial when achieved.
    """

    def __init__(self, threshold):
        if threshold < 0:
            raise ValueError(f"Threshold cannot be negative but got {threshold}.")

        self._threshold = threshold
        self._consecutive_pruned_count = 0

    def __call__(self, study, trial):
        """
        Stops study when consecutive prune count exceeds threshold.

        Callback function that gets called after each trial.

        Parameters
        ----------
        study : optuna.study.Study
            Study object of the target study.
        trial : optuna.trial.FrozenTrial
            FrozenTrial object of the target trial. Take a copy before modifying this
            object.
        """
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consecutive_pruned_count += 1
        else:
            self._consecutive_pruned_count = 0

        if self._consecutive_pruned_count > self._threshold:
            study.stop()


class _StopStudyAfterNPruning:
    """
    Optuna study callback that stops the study after begin N times pruned.

    Parameters
    ----------
    threshold : int
        Critical threshold for total number of pruned trials. Stops trial when achieved.
    """

    def __init__(self, threshold):
        if threshold < 0:
            raise ValueError(f"Threshold cannot be negative but got {threshold}.")

        self._threshold = threshold

    def __call__(self, study, trial):
        """
        Stops study when total prune count exceeds threshold.

        Callback function that gets called after each trial.

        Parameters
        ----------
        study : optuna.study.Study
            Study object of the target study.
        trial : optuna.trial.FrozenTrial
            FrozenTrial object of the target trial. Take a copy before modifying this
            object.
        """
        all_trials = study.get_trials(deepcopy=False)
        n_prunings = len(
            [t for t in all_trials if t.state == optuna.trial.TrialState.PRUNED]
        )

        if n_prunings > self._threshold:
            study.stop()
