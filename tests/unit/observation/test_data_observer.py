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
        monotonic_incr = 4.2 * np.arange(-10, size - 10)[:, None]  # start=-10, step=4.2
        monotonic_decr = 6.3 * np.arange(-3, size - 3)[::-1, None]
        constants = np.zeros(size)[:, None]
        random = np.random.normal(size=size)[:, None]
        x = np.concatenate([monotonic_incr, monotonic_decr, constants, random], axis=1)
        y = random.reshape(-1)  # does not matter

        # Add nan to first and random
        x[1, 0] = np.nan

        # Observe
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline._read_data(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_monotonic_columns()
        msg = str(record[0].message)
        monotonic_cols = json.loads(re.search(r"\[.*]", msg).group(0))
        assert set(monotonic_cols) == {0, 1}, "Wrong monotonic columns identified."

    def test_minority_sensitivity(self):
        # Setup
        x = np.hstack(
            (
                np.random.normal(size=(100, 1)),
                np.concatenate((np.zeros((2, 1)), np.random.normal(100, 1, (98, 1)))),
            )
        )
        y = np.concatenate((np.zeros(5), np.ones(95)))

        # Add nan
        x[1, 0] = np.nan

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

        # Add nan
        x[0, 0] = np.nan

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

        # Add nan
        x[0, 0] = np.nan

        # Observe
        pipeline = Pipeline(grid_search_iterations=0)
        pipeline._read_data(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_extreme_values()
        msg = str(record[0].message)
        extreme_cols = json.loads(re.search(r"\[.*]", msg).group(0))
        assert extreme_cols == [1], "Wrong minority sensitive columns identified."
