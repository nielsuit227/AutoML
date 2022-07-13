#  Copyright (c) 2022 by Amplo.

import multiprocessing as mp
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV  # noqa

from amplo.grid_search._base import BaseGridSearch
from amplo.utils import deprecated


@deprecated("Please use Optuna, not experimental and more efficient.")
class HalvingGridSearch(BaseGridSearch):
    def __init__(self, model, *args, **kwargs):
        if "timeout" in kwargs:
            warnings.warn("Parameter `timeout` has no effect")
            kwargs.pop("timeout")

        super().__init__(model, *args, **kwargs)

        # Halving settings
        self.resource = "n_samples"
        self.max_resource = "auto"
        self.min_resource = 200
        self.set_resources()

    def set_resources(self):
        if "CatBoost" in type(self.model).__name__:
            self.resource = "n_estimators"
            self.max_resource = 3000
            self.min_resource = 250
        if (
            self.model.__module__ == "sklearn.ensemble._bagging"
            or self.model.__module__ == "xgboost.sklearn"
            or self.model.__module__ == "lightgbm.sklearn"
            or self.model.__module__ == "sklearn.ensemble._forest"
        ):
            self.resource = "n_estimators"
            self.max_resource = 1500
            self.min_resource = 50

    def _get_hyper_params(self):
        param_values = self._hyper_parameter_values
        param_values.pop(
            "CONDITIONALS", {}
        )  # drop conditionals as they are not supported
        param_values.pop(
            "n_jobs", None
        )  # drop n_jobs as HalvingGridSearch also uses this parameter
        param_values.pop(
            "n_estimators", None
        )  # otherwise cannot use parameter n_estimators as the resource

        # Drop/adjust some more parameter names which HalvinGridSearch cannot handle
        model_name = type(self.model).__name__
        if model_name in ("BaggingClassifier", "BaggingRegressor"):
            # Apparently the regressor cannot handle any features for halving
            return {}
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

            # Sanity checks
            assert (
                len(p_args) == 2 or p_type == "categorical"
            ), "Only categorical parameter can have more/less than two suggest args"

            # Suggest parameter given the arguments
            if p_type == "categorical":
                params[p_name] = p_args
            elif p_type == "int":
                params[p_name] = randint(*p_args)
            elif p_type == "logint":
                raise NotImplementedError("logint is not yet implemented")
            elif p_type == "uniform":
                params[p_name] = uniform(*p_args)
            elif p_type == "loguniform":
                params[p_name] = loguniform(*p_args)
            else:
                raise NotImplementedError("Invalid parameter specification")

        return params

    def fit(self, x, y):
        # Update minimum resource for samples (based on dataset)
        if self.resource == "n_samples":
            self.min_resource = int(0.2 * len(x)) if len(x) > 5000 else len(x)

        # Parameters
        if len(np.unique(y)) == 2:
            self.binary = True
        self.samples = len(y)
        if self.params is None:
            self.params = self._get_hyper_params()

        # Set up and run grid search
        halving_random_search = HalvingRandomSearchCV(
            self.model,
            self.params,
            n_candidates=self.n_trials,
            resource=self.resource,
            max_resources=self.max_resource,
            min_resources=self.min_resource,
            cv=self.cv,
            scoring=self.scoring,
            factor=3,
            n_jobs=mp.cpu_count() - 1,
            verbose=self.verbose,
        )
        halving_random_search.fit(x, y)

        # Parse results
        scikit_results = pd.DataFrame(halving_random_search.cv_results_)
        results = pd.DataFrame(
            {
                "date": datetime.today().strftime("%d %b %y"),
                "model": type(self.model).__name__,
                "params": scikit_results["params"],
                "mean_objective": scikit_results["mean_test_score"],
                "std_objective": scikit_results["std_test_score"],
                "worst_case": (
                    scikit_results["mean_test_score"] - scikit_results["std_test_score"]
                ),
                "mean_time": scikit_results["mean_fit_time"],
                "std_time": scikit_results["std_fit_time"],
            }
        )

        # Update resource in results
        if self.resource != "n_samples":
            for i in range(len(results)):
                results.loc[results.index[i], "params"][
                    self.resource
                ] = self.max_resource

        return results
