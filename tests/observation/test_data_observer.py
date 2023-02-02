#  Copyright (c) 2022 by Amplo.

import ast
import re

import numpy as np
import pandas as pd
import pytest

from amplo.automl import DataProcessor
from amplo.observation._base import ProductionWarning
from amplo.observation.data import DataObserver
from tests import find_first_warning_of_type


class TestDataObserver:
    @pytest.mark.parametrize("index_type", ["single", "multi"])
    def test_monotonic_columns(self, index_type):
        size = 100
        monotonic_incr = 4.2 * np.arange(-10, size - 10)
        monotonic_decr = 6.3 * np.arange(-3, size - 3)[::-1]
        constants = np.zeros(size)
        random = np.random.normal(size=size)

        # Prepare data
        y1 = random.reshape(-1)  # does not matter
        if index_type == "single":
            data = pd.DataFrame(
                {
                    "inc": monotonic_incr,
                    "dec": monotonic_decr,
                    "con": constants,
                    "random": random,
                    "y1": y1,
                }
            )
        elif index_type == "multi":
            data = pd.DataFrame(
                {
                    "inc": np.concatenate([monotonic_incr, monotonic_incr], axis=0),
                    "dec": np.concatenate([monotonic_decr, monotonic_decr], axis=0),
                    "con": np.concatenate([constants, constants], axis=0),
                    "random": np.concatenate([random, random], axis=0),
                    "y1": np.concatenate([y1, y1], axis=0),
                },
                index=pd.MultiIndex.from_product([[0, 1], range(size)]),
            )
        else:
            raise ValueError("Invalid index_type.")

        # Add nan to first and random
        data.iloc[1, 0] = np.nan

        # Observe)
        with pytest.warns(ProductionWarning) as record:
            DataObserver().check_monotonic_columns(data)
        msg = str(find_first_warning_of_type(ProductionWarning, record).message)
        monotonic_cols = ast.literal_eval(re.search(r"\[.*]", msg).group(0))  # type: ignore[union-attr]
        assert set(monotonic_cols) == {
            "dec",
            "inc",
        }, "Wrong monotonic columns identified."

    def test_minority_sensitivity(self):
        # Setup
        data = pd.DataFrame(
            {
                "random": np.random.normal(size=100),
                "x": np.concatenate((np.zeros(2), np.random.normal(100, 1, 98))),
                "y": np.concatenate((np.zeros(5), np.ones(95))),
            }
        )
        data.iloc[1, 0] = np.nan

        # Observe
        with pytest.warns(ProductionWarning) as record:
            DataObserver().check_minority_sensitivity(data, target="y")
        msg = str(find_first_warning_of_type(ProductionWarning, record).message)
        sensitive_cols = ast.literal_eval(re.search(r"\[.*]", msg).group(0))  # type: ignore[union-attr]
        assert sensitive_cols == ["x"], "Wrong minority sensitive columns identified."

    def test_categorical_mismatch(self):
        """
        Detects whether there is an issue with categorical variables being too similar.

        Note: it does not detect "New York" vs. "new-york", as this is taken care off by
        cleaning feature names.
        """
        # Setup
        data = pd.DataFrame(
            {
                "cat_a": np.array(["New York"] * 50 + ["new-york"] * 50),
                "cat_b": np.array(["Something"] * 50 + ["Somethign"] * 50),
                "normal": np.random.normal(100, 5, 100),
                "y": np.concatenate((np.zeros(5), np.ones(95))),
            }
        )

        # Add nan
        data.iloc[0, 0] = np.nan

        # Observe
        dp = DataProcessor()
        dp.fit_transform(data)
        with pytest.warns(ProductionWarning) as record:
            DataObserver().check_categorical_mismatch(dp.dummies_)
        msg = str(find_first_warning_of_type(ProductionWarning, record).message)
        sensitive_cols = ast.literal_eval(re.search(r"\[.*]", msg).group(0))  # type: ignore[union-attr]
        assert (
            len(sensitive_cols) == 1 and "cat_b" in sensitive_cols[0]
        ), "Wrong categorical mismatch columns identified."

    def test_extreme_values(self):
        # Setup
        data = pd.DataFrame(
            {
                "normal": np.random.normal(size=100),
                "linear": np.linspace(1000, 10000, 100),
                "y": np.concatenate((np.zeros(5), np.ones(95))),
            }
        )

        # Add nan
        data.iloc[0, 0] = np.nan

        # Observe
        with pytest.warns(ProductionWarning) as record:
            DataObserver().check_extreme_values(data)
        msg = str(find_first_warning_of_type(ProductionWarning, record).message)
        extreme_cols = ast.literal_eval(re.search(r"\[.*]", msg).group(0))  # type: ignore[union-attr]
        assert extreme_cols == [
            "linear"
        ], "Wrong minority sensitive columns identified."

    def test_label_issues(self):
        # Setup
        data = pd.DataFrame(
            {
                "x": np.hstack(
                    (
                        np.random.normal(-2, 0.1, size=50),
                        np.random.normal(2, 0.1, size=50),
                    )
                ),
                "y": np.hstack((np.ones(50), np.zeros(49), 1)),
            }
        )

        # Observe
        with pytest.warns(ProductionWarning) as record:
            DataObserver().check_label_issues(data, mode="classification", target="y")
        msg = str(find_first_warning_of_type(ProductionWarning, record).message)
        label_issues = ast.literal_eval(re.search(r"\[.*]", msg).group(0))  # type: ignore
        assert label_issues == [99], "Wrong label issues identified."
