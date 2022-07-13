#  Copyright (c) 2022 by Amplo.

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from amplo.automl import Modeller
from tests import rmtree


@pytest.fixture(scope="class", params=["classification", "regression"])
def make_mode(request):
    mode = request.param
    if mode == "classification":
        x, y = make_classification()
        objective = "neg_log_loss"
    elif mode == "regression":
        x, y = make_regression()
        objective = "r2"
    else:
        raise ValueError("Invalid mode")
    request.cls.mode = mode
    request.cls.objective = objective
    request.cls.x = pd.DataFrame(x)
    request.cls.y = pd.Series(y)
    yield


@pytest.fixture(scope="class", autouse=True)
def clean():
    yield None
    rmtree("tmp/")


@pytest.mark.usefixtures("make_mode")
class TestModelling:
    folder = "tmp/"

    def test_modeller(self):
        mod = Modeller(mode=self.mode, objective=self.objective, folder=self.folder)
        mod.fit(self.x, self.y)
        # Tests
        assert isinstance(
            mod.results, pd.DataFrame
        ), "Results should be type pd.DataFrame"
        assert len(mod.results) != 0, "Results empty"
        assert mod.results["mean_objective"].max() < 1, "R2 needs to be smaller than 1"
        assert (
            not mod.results["mean_objective"].isna().any()
        ), "Mean Objective shouldn't contain NaN"
        assert (
            not mod.results["std_objective"].isna().any()
        ), "Std Objective shouldn't contain NaN"
        assert (
            not mod.results["mean_time"].isna().any()
        ), "Mean time shouldn't contain NaN"
        assert (
            not mod.results["std_time"].isna().any()
        ), "Std time shouldn't contain NaN"
        assert "date" in mod.results.keys()
        assert "model" in mod.results.keys()
        assert "dataset" in mod.results.keys()
        assert "params" in mod.results.keys()

    @pytest.mark.parametrize("n_samples", [100, 100_000])
    def test_return(self, n_samples):
        Modeller(mode=self.mode, samples=n_samples).return_models()
