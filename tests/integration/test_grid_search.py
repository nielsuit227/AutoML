#  Copyright (c) 2022 by Amplo.
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold

from amplo.automl.grid_search import OptunaGridSearch
from amplo.utils.io import parse_json
from tests import get_all_modeller_models


@pytest.fixture(scope="class", params=["regression", "classification"])
def make_data(request):
    mode = request.param

    if mode == "regression":
        request.cls.k_fold = KFold
        request.cls.objective = "r2"
        x, y = make_regression(n_features=5)  # type: ignore
    elif mode == "classification":
        request.cls.k_fold = StratifiedKFold
        request.cls.objective = "neg_log_loss"
        x, y = make_classification(n_features=5)
    else:
        raise ValueError("Mode is invalid")

    request.cls.mode = mode
    request.cls.data = pd.DataFrame(x)
    request.cls.data["target"] = y
    yield


@pytest.mark.usefixtures("make_data")
class TestGridSearch:
    objective: str
    mode: str
    k_fold: type[KFold] | type[StratifiedKFold]
    data: pd.DataFrame

    def test_each_model(self):
        """
        Given the models which depend on 1) mode and 2) num_samples,
        each model is trained on the given grid search class.

        Thanks to the fact that all grid search models have the same interface,
        this is possible :-)
        """
        models = get_all_modeller_models(mode=self.mode, objective=self.objective)

        for model in models:
            # Grid search
            search = OptunaGridSearch(
                model,
                target="target",
                cv=self.k_fold(n_splits=3),
                verbose=0,
                timeout=10,
                n_trials=2,
                scoring=self.objective,
            )
            results = search.fit(self.data)

            # Tests
            model_name = type(model).__name__
            assert isinstance(results, pd.DataFrame), (
                f"Expected result to be pandas.DataFrame"
                f"but found {type(results)} in {model_name} instead"
            )
            results_keys = [
                "worst_case",
                "score",
                "params",
                "time",
            ]
            assert all(
                key in results.keys() for key in results_keys
            ), f"Keys are missing in {model_name}"
            assert len(results) > 0, f"Results are empty in {model_name}"
            model.set_params(**parse_json(results.iloc[0]["params"]))
