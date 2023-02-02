#  Copyright (c) 2022 by Amplo.
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from amplo.automl import Modeller
from tests import rmtree


@pytest.fixture(scope="class", autouse=True)
def clean():
    yield None
    rmtree("tmp/")


@pytest.mark.parametrize("mode", ["classification", "regression"])
class TestModelling:
    folder = "tmp/"
    mode: str
    data: pd.DataFrame

    def objective(self, mode: str) -> str:
        if mode == "classification":
            return "neg_log_loss"
        else:
            return "neg_mean_squared_error"

    def test_modeller(self, mode, data):
        with mock.patch(
            "sklearn.model_selection.cross_val_score",
            return_value=[-0.1, -0.1, -0.1, -0.1, -0.1],
        ):

            # Limit data
            data = data[:30]

            mod = Modeller(target="target", mode=mode, objective=self.objective(mode))
            mod.fit(data)
            # Tests
            assert isinstance(mod.results, list), "Results should be type list"
            assert len(mod.results) != 0, "Results empty"
            assert max(r.score for r in mod.results) < 0, "neg_log_loss or neg_mse > 0"
            assert not any(
                np.isnan(r.worst_case) for r in mod.results
            ), "worst_case shouldn't contain NaN"
            assert not any(
                np.isnan(r.time) for r in mod.results
            ), "Time shouldn't contain NaN"
            assert hasattr(mod.results[0], "date")
            assert hasattr(mod.results[0], "model")
            assert hasattr(mod.results[0], "params")

    @pytest.mark.parametrize("n_samples", [100, 100_000])
    def test_return(self, mode, n_samples):
        Modeller(mode=mode, samples=n_samples).return_models()
