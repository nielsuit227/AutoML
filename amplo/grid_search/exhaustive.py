#  Copyright (c) 2022 by Amplo.

import copy
import itertools
import random
import re
import time
from datetime import datetime
from typing import Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from amplo.grid_search._base import BaseGridSearch


class ExhaustiveGridSearch(BaseGridSearch):
    """
    Basic exhaustive grid search.

    Parameters
    ----------
    model : Amplo.AutoML.Modeller.ModelType
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

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        # Initiate
        self.parsed_params = []
        self.result = []

    def _get_hyper_params(self) -> Dict[str, Union[None, bool, int, float, str, list]]:
        param_values = self._hyper_parameter_values
        # Drop conditionals as they are not supported
        param_values.pop("CONDITIONALS", None)

        # Drop/adjust some more parameter names which HalvingGridSearch cannot handle
        model_name = type(self.model).__name__
        if model_name in ("BaggingClassifier", "BaggingRegressor"):
            param_values["max_features"] = ("categorical", [0.5, 0.8, 0.9, 1.0], None)
        elif model_name in ("LGBMRegressor", "LGBMClassifier"):
            # Someway it cannot handle uniform distributions for this two cases
            param_values["bagging_fraction"] = (
                "categorical",
                [0.5, 0.7, 0.9, 1.0],
                None,
            )
            param_values["feature_fraction"] = (
                "categorical",
                [0.5, 0.7, 0.9, 1.0],
                None,
            )

        # Extract parameter distributions
        params = {}
        for p_name, value in param_values.items():

            # Read out
            p_type = value[0]  # parameter type (str)
            p_args = value[1]  # parameter arguments (list, tuple)
            p_count = (
                value[2] if p_type != "categorical" else None
            )  # n_samples for np.linspace or np.logspace

            # Sanity checks
            assert (
                len(p_args) == 2 or p_type == "categorical"
            ), "Only categorical parameter can have more/less than two suggest args"

            # Skip parameters when none are wanted
            if p_count == 0:
                continue
            # Suggest parameter given the arguments
            if p_type == "categorical":
                params[p_name] = p_args
                continue
            elif p_type in ("int", "uniform"):
                params[p_name] = np.linspace(*p_args, p_count)
            elif p_type in ("logint", "loguniform"):
                p_args = [np.log(arg) for arg in p_args]  # logarithmise for np.logspace
                params[p_name] = np.logspace(*p_args, p_count)
            else:
                raise NotImplementedError("Invalid parameter specification")
            # Discretize and convert to list
            if "int" in p_type:
                params[p_name] = params[p_name].astype("int")
            params[p_name] = np.unique(params[p_name]).tolist()

        return params

    def _parse_params(self):
        if len(self.params.items()) > 0:
            k, v = zip(*self.params.items())
            self.parsed_params = [
                dict(zip(k, v)) for v in itertools.product(*self.params.values())
            ]
            random.shuffle(self.parsed_params)
            self.logger.info(
                f"[GridSearch] Optimizing {type(self.model).__name__}, "
                f"{self.cv.n_splits}-folds with {len(self.parsed_params)} parameter "
                f"combinations and {len(self.parsed_params) * self.cv.n_splits} runs."
            )
        else:
            self.parsed_params = [{}]

    def fit(self, x, y):
        start_time = time.time()
        # todo check data
        # Parse Params
        model_name = type(self.model).__name__
        is_classification = bool(
            re.match(r".*(Classification|Classifier|SVC)", model_name)
        )
        if is_classification:
            if len(np.unique(y)) == 2:
                self.binary = True
            else:
                self.binary = False
        # Convert to Numpy
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = np.array(x)
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = np.array(y).reshape((-1))
        self.samples = len(y)

        # Get params
        if self.params is None:
            self.params = self._get_hyper_params()
        self._parse_params()

        # Loop through parameters
        for i, param in tqdm(enumerate(self.parsed_params)):
            # Timeout or trials limit
            if time.time() - start_time > self.timeout or i > self.n_trials:
                return pd.DataFrame(self.result)

            # Scoring and timing vector
            scoring = []
            timing = []

            # Cross Validation
            for train_ind, val_ind in self.cv.split(x, y):
                # Start Timer
                t = time.time()

                # Split data
                x_train, x_val = x[train_ind], x[val_ind]
                y_train, y_val = y[train_ind], y[val_ind]

                # Model training
                model = copy.deepcopy(self.model)
                model.set_params(**param)
                model.fit(x_train, y_train)

                # Results
                scoring.append(self.scoring(model, x_val, y_val))
                timing.append(time.time() - t)

            # Output Printing
            self.logger.info(
                f"[GridSearch][{datetime.now().strftime('%H:%M')}] "
                f"Score: {np.mean(scoring):.4f} \u00B1 {np.std(scoring):.4f} "
                f"(in {np.mean(timing):.1f} seconds)."
            )

            self.result.append(
                {
                    "date": datetime.today().strftime("%d %b %y"),
                    "model": type(model).__name__,
                    "mean_objective": np.mean(scoring),
                    "std_objective": np.std(scoring),
                    "worst_case": np.mean(scoring) - np.std(scoring),
                    "params": param,
                    "mean_time": np.mean(timing),
                    "std_time": np.std(timing),
                }
            )
        return pd.DataFrame(self.result)
