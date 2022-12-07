#  Copyright (c) 2022 by Amplo.

import pandas as pd
import pytest

from amplo.automl import Modeller
from tests import rmtree


@pytest.fixture(scope="class", autouse=True)
def clean():
    yield None
    rmtree("tmp/")


@pytest.mark.usefixtures("make_mode")
class TestModelling:
    folder = "tmp/"
    mode: str
    objective: str
    data: pd.DataFrame

    def test_modeller(self):
        mod = Modeller(target="target", mode=self.mode, objective=self.objective)
        mod.fit(self.data)
        # Tests
        assert isinstance(
            mod.results, pd.DataFrame
        ), "Results should be type pd.DataFrame"
        assert len(mod.results) != 0, "Results empty"
        assert mod.results["score"].max() < 1, "R2 needs to be smaller than 1"
        assert (
            not mod.results["worst_case"].isna().any()
        ), "worst_case shouldn't contain NaN"
        assert not mod.results["time"].isna().any(), "Time shouldn't contain NaN"
        assert "date" in mod.results.keys()
        assert "model" in mod.results.keys()
        assert "dataset" in mod.results.keys()
        assert "params" in mod.results.keys()

    @pytest.mark.parametrize("n_samples", [100, 100_000])
    def test_return(self, n_samples):
        Modeller(mode=self.mode, samples=n_samples).return_models()
