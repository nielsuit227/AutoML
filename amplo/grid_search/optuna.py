#  Copyright (c) 2022 by Amplo.

import copy
import time
import warnings
from datetime import datetime
from typing import Dict, Union

import numpy as np
import optuna
import pandas as pd

from amplo.grid_search._base import BaseGridSearch

__all__ = ["OptunaGridSearch"]


class OptunaGridSearch(BaseGridSearch):
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
    scoring : str or sklearn.metrics._scorer._BaseScorer
        A valid string for `sklearn.metrics.get_scorer`.
    verbose : int
        Verbose logging.
    """

    def __init__(self, model, *args, **kwargs):
        if "params" in kwargs:
            warnings.warn("Parameter `params` has no effect")
            kwargs.pop("params")

        super().__init__(model, *args, **kwargs)

        # Counter to keep track of number of trials
        self.trial_count = -1

    @staticmethod
    def _get_suggestion(trial, p_name, p_value):
        """
        Get suggestion for next hyperparameter.

        Parameters
        ----------
        trial : optuna.Trial
            The trial to sample a suggestion from it.
        p_name : str
            Name of the hyperparameter.
        p_value : (str, list) or (str, list, int)
            Sampling information for hyperparameter (dtype, range).

        Returns
        -------
        None or bool or int or float or str
            A specific parameter for given hyperparameter.
        """

        # Read out
        p_type = p_value[0]  # parameter type (str)
        p_args = p_value[1]  # parameter arguments (list, tuple)
        p_kwargs = {}  # additional kwargs parameter, depends on parameter type

        # Sanity checks
        assert (
            len(p_args) == 2 or p_type == "categorical"
        ), "Only categorical parameter can have more/less than two suggest args"

        # Find trial.suggest_{...} type
        if p_type == "categorical":
            p_args = [p_args]  # enclose since will be expanded afterwards
            suggest = trial.suggest_categorical
        elif p_type == "int":
            suggest = trial.suggest_int
        elif p_type == "logint":
            p_kwargs += dict(log=True)
            suggest = trial.suggest_int
        elif p_type == "uniform":
            suggest = trial.suggest_uniform
        elif p_type == "loguniform":
            suggest = trial.suggest_loguniform
        else:
            raise NotImplementedError("Invalid parameter specification")

        # Suggest parameter given the arguments
        return suggest(p_name, *p_args)

    def _get_hyper_params(self, trial) -> Dict[str, Union[None, bool, int, float, str]]:
        """Use trial to sample from available grid search parameters

        Parameters
        ----------
        trial : optuna.Trial
            The trial from optuna study.

        Returns
        -------
        params : dict of {str : None or bool or int or float or str}
            Sampled grid search parameters.
        """

        param_values = self._hyper_parameter_values
        conditionals = param_values.pop("CONDITIONALS", {})
        params = {}

        # Add non-conditional parameter suggestions (don't depend on another)
        for name, value in param_values.items():
            params[name] = self._get_suggestion(trial, name, value)

        # Add conditional parameter suggestions that satisfy condition:
        for check_p_name, check_p_criteria in conditionals.items():
            for matching_value, additional_params in check_p_criteria:
                # Add parameters only if criterion is satisfied
                if matching_value != params.get(check_p_name, None):
                    # No match
                    continue
                for name, value in additional_params.items():
                    params[name] = self._get_suggestion(trial, name, value)

        return params

    def fit(self, x, y):
        if isinstance(y, pd.DataFrame):
            assert len(y.keys()) == 1, "Multiple target columns not supported."
            y = y[y.keys()[0]]
        assert isinstance(x, pd.DataFrame), "X should be Pandas DataFrame"
        assert isinstance(y, pd.Series), "Y should be Pandas Series or DataFrame"

        # Set mode
        self.binary = y.nunique() == 2
        self.samples = len(y)

        # Store
        self.x, self.y = x, y

        # Set up study
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=236868),
            direction="maximize",
            pruner=_BadTrialPruner(2.0, 15),
        )
        study.optimize(
            self.objective,
            timeout=self.timeout,
            n_trials=self.n_trials,
            callbacks=[
                _StopStudyWhenConsecutivePruning(10),
                _StopStudyAfterNPruning(50),
            ],
        )

        # Parse results
        optuna_results = study.trials_dataframe()
        results = pd.DataFrame(
            {
                "date": datetime.today().strftime("%d %b %y"),
                "model": type(self.model).__name__,
                "params": [x.params for x in study.get_trials()],
                "mean_objective": optuna_results["value"],
                "std_objective": optuna_results["user_attrs_std_value"],
                "worst_case": optuna_results["value"]
                - optuna_results["user_attrs_std_value"],
                "mean_time": optuna_results["user_attrs_mean_time"],
                "std_time": optuna_results["user_attrs_std_time"],
            }
        )
        self.trial_count = len(study.trials)

        return results

    def objective(self, trial: optuna.Trial):
        # Metrics
        scores = []
        times = []
        master = copy.deepcopy(self.model)

        # Cross validation
        for step, (t, v) in enumerate(self.cv.split(self.x, self.y)):
            # Split data
            xt, yt = self.x.iloc[t], self.y.iloc[t]
            xv, yv = self.x.iloc[v], self.y.iloc[v]

            # Train model
            t_start = time.time()
            model = copy.deepcopy(master)
            model.set_params(**self._get_hyper_params(trial))
            model.fit(xt, yt)

            # Results
            score = self.scoring(model, xv, yv)
            scores.append(score)
            times.append(time.time() - t_start)

            # Report intermediate values
            trial.report(score, step)

            # Stop cross validation when it is not promising
            if trial.should_prune():
                break

        # Set manual metrics
        mean_score = np.mean(scores)
        trial.set_user_attr("mean_time", np.mean(times))
        trial.set_user_attr("std_time", np.std(times))
        trial.set_user_attr("mean_value", np.mean(scores))
        trial.set_user_attr("std_value", np.std(scores))

        # Stop trial (avoid overwriting)
        if trial.number == self.n_trials:
            trial.study.stop()

        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned

        return mean_score


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

    def prune(self, study, trial):
        # Don't prune while startup trials
        all_trials = study.get_trials(deepcopy=False)
        n_trials = len(
            [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        if n_trials < self._n_startup_trials:
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
