#  Copyright (c) 2022 by Amplo.

import json
import re

import numpy as np
import pandas as pd
import pytest

from amplo import Pipeline
from amplo.observation._base import ProductionWarning
from amplo.observation.data import DataObserver
from tests import find_first_warning_of_type


class TestDataObserver:
    @pytest.mark.parametrize("index_type", ["single", "multi"])
    def test_monotonic_columns(self, index_type):
        size = 100
        monotonic_incr = 4.2 * np.arange(-10, size - 10)[:, None]  # start=-10, step=4.2
        monotonic_decr = 6.3 * np.arange(-3, size - 3)[::-1, None]
        constants = np.zeros(size)[:, None]
        random = np.random.normal(size=size)[:, None]

        # Prepare data
        x1 = np.concatenate([monotonic_incr, monotonic_decr, constants, random], axis=1)
        x2 = np.concatenate(4 * [random], axis=1)
        y1 = random.reshape(-1)  # does not matter
        if index_type == "single":
            x = pd.DataFrame(x1)
            y = pd.Series(y1)
        elif index_type == "multi":
            x = pd.DataFrame(np.concatenate([x1, x2], axis=0))
            x.index = pd.MultiIndex.from_product([[0, 1], range(size)])
            y = pd.Series(np.concatenate([y1, y1], axis=0))
            y.index = x.index
        else:
            raise ValueError("Invalid index_type.")

        # Add nan to first and random
        x.iloc[1, 0] = np.nan

        # Observe
        pipeline = Pipeline(n_grid_searches=0)
        pipeline._read_data(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_monotonic_columns()
        msg = str(find_first_warning_of_type(ProductionWarning, record).message)
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
        pipeline = Pipeline(n_grid_searches=0)
        pipeline._read_data(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_minority_sensitivity()
        msg = str(find_first_warning_of_type(ProductionWarning, record).message)
        sensitive_cols = json.loads(re.search(r"\[.*]", msg).group(0))
        assert sensitive_cols == [1], "Wrong minority sensitive columns identified."

    def test_categorical_mismatch(self):
        # Setup
        x = np.hstack(
            (
                np.array(["New York"] * 50 + ["new-york"] * 50).reshape((-1, 1)),
                np.array(["Something"] * 50 + ["Somethign"] * 50).reshape((-1, 1)),
                np.random.normal(100, 5, (100, 1)),
            )
        )
        y = np.concatenate((np.zeros(5), np.ones(95)))

        # Add nan
        x[0, 0] = np.nan

        # Observe
        pipeline = Pipeline(n_grid_searches=0, cv_splits=3)
        pipeline.data_preparation(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_categorical_mismatch()
        msg = str(find_first_warning_of_type(ProductionWarning, record).message)
        sensitive_cols = json.loads(re.search(r"\[.*]", msg).group(0))
        assert sensitive_cols == [
            {"feature_1": ["something", "somethign"]}
        ], "Wrong categorical mismatch columns identified."

    def test_extreme_values(self):
        # Setup
        x = np.vstack((np.random.normal(size=100), np.linspace(1000, 10000, 100))).T
        y = np.concatenate((np.zeros(5), np.ones(95)))

        # Add nan
        x[0, 0] = np.nan

        # Observe
        pipeline = Pipeline(n_grid_searches=0)
        pipeline._read_data(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_extreme_values()
        msg = str(find_first_warning_of_type(ProductionWarning, record).message)
        extreme_cols = json.loads(re.search(r"\[.*]", msg).group(0))
        assert extreme_cols == [1], "Wrong minority sensitive columns identified."

    def test_label_issues(self):
        # Setup
        x = np.hstack(
            (np.random.normal(-2, 0.1, size=50), np.random.normal(2, 0.1, size=50))
        )
        y = np.hstack((np.ones(50), np.zeros(49), 1))

        # Observe
        pipeline = Pipeline(n_grid_searches=0)
        pipeline.fit(x, y)
        obs = DataObserver(pipeline=pipeline)
        with pytest.warns(ProductionWarning) as record:
            obs.check_label_issues()
        msg = str(find_first_warning_of_type(ProductionWarning, record).message)
        label_issues = json.loads(re.search(r"\[.*]", msg).group(0))
        assert label_issues == [99], "Wrong label issues identified."
