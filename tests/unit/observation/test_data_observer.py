import json
import re

import numpy as np
import pytest

from Amplo import Pipeline
from Amplo.Observation._data_observer import DataObserver
from Amplo.Observation.base import ProductionWarning


class TestDataObserver:
    def test_monotonic_columns(self):
        size = 100
        monotonic = np.array(range(-10, size - 10)) * 4.2  # start=-10, step=4.2
        random = np.random.normal(size=size)
        x = np.concatenate([monotonic[:, None], random[:, None]], axis=1)
        y = random  # does not matter

        # Observe
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline._read_data(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_monotonic_columns()
        msg = str(record[0].message)
        monotonic_cols = json.loads(re.search(r"\[.*]", msg).group(0))
        assert monotonic_cols == [0], "Wrong monotonic columns identified."

    def test_minority_sensitivity(self):
        # Setup
        x = np.hstack(
            (
                np.random.normal(size=(100, 1)),
                np.concatenate((np.zeros((2, 1)), np.random.normal(100, 5, (98, 1)))),
            )
        )
        y = np.concatenate((np.zeros(5), np.ones(95)))

        # Observe
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline._read_data(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_minority_sensitivity()
        msg = str(record[0].message)
        sensitive_cols = json.loads(re.search(r"\[.*]", msg).group(0))
        assert sensitive_cols == [1], "Wrong minority sensitive columns identified."

    def test_categorical_mismatch(self):
        # Setup
        x = np.hstack(
            (
                np.array(["New York"] * 50 + ["new-york"] * 50).reshape((-1, 1)),
                np.random.normal(100, 5, (100, 1)),
            )
        )
        y = np.concatenate((np.zeros(5), np.ones(95)))

        # Observe
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline._read_data(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_categorical_mismatch()
        msg = str(record[0].message)
        sensitive_cols = json.loads(re.search(r"\[.*]", msg).group(0))
        assert sensitive_cols == [
            {"0": ["new-york", "New York"]}
        ], "Wrong categorical mismatch columns identified."

    def test_extreme_values(self):
        # Setup
        x = np.vstack((np.random.normal(size=100), np.linspace(1000, 10000, 100))).T
        y = np.concatenate((np.zeros(5), np.ones(95)))

        # Observe
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline._read_data(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_extreme_values()
        msg = str(record[0].message)
        extreme_cols = json.loads(re.search(r"\[.*]", msg).group(0))
        assert extreme_cols == [1], "Wrong minority sensitive columns identified."
