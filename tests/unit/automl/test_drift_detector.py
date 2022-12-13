#  Copyright (c) 2022 by Amplo.

import json
import warnings

import numpy as np
import pytest
import scipy.stats

from amplo.automl import DriftDetector
from amplo.automl.drift_detection import DataDriftWarning
from amplo.utils.testing import DummyDataSampler, make_num_data


class DummyPredictor(DummyDataSampler):
    def predict(self, data):
        return self.sample_data(len(data))


class TestDriftDetector:
    def test_distribution_fits(self):
        # Setup
        ref, cols = make_num_data(500)
        test = ref.iloc[np.random.permutation(len(ref))[:10]]
        drift = DriftDetector(target="target")
        drift.fit(ref)

        # Checks
        assert len(drift.check(test)) == 0, "Test data found inconsistent"
        assert len(drift.check(ref.max() + 1)) == len(
            ref.columns
        ), "Maxima not detected"
        assert len(drift.check(ref.min() - 1)) == len(
            ref.columns
        ), "Minima not detected"

    def test_storable(self):
        df, cols = make_num_data(100)
        drift = DriftDetector(target="target")
        drift.fit(df)
        json.dumps(drift.bins_)
        pred = np.random.randint(0, 2, (100,))
        old = drift.add_output_bins((), pred)
        drift.add_output_bins(old, pred)

    def test_drift_warning(self):
        """
        Ensure that minor changes in data do not trigger warnings.
        """
        # Create dummy data --
        data_1, _ = make_num_data(500, num_dists=["uniform", "norm"])
        data_2, _ = make_num_data(10, num_dists=["uniform", "norm"])
        data_2["num_0"] += 15
        data_2["num_1"] -= 10
        # NOTE: categorical data is not compared

        # Create dummy predictors
        dummy_model_1 = DummyPredictor()
        dummy_model_2 = DummyPredictor(scipy.stats.randint(0, 10))  # type: ignore

        # Instantiate and fit drift detector
        drift = DriftDetector(target="target", n_bins=10)
        drift.fit(data_1)
        drift.fit_output(dummy_model_1, data_1)  # type: ignore

        # Assert that no DataDriftWarning occurs when given same data and model
        with warnings.catch_warnings(record=True) as caught_warnings:
            # Check drift on input
            drift.check(data_1)
            # Check drift on output
            drift.check_output(dummy_model_1, data_1)  # type: ignore
        if any(
            issubclass(warning.category, DataDriftWarning)
            for warning in caught_warnings
        ):
            raise AssertionError("Unnecessary DataDriftWarning detected")

        # Assert that DataDriftWarning occurs when given new data
        with pytest.warns(DataDriftWarning):
            drift.check(data_2)

        # Assert that DataDriftWarning occurs when given new model
        with pytest.warns(DataDriftWarning):
            drift.check_output(dummy_model_2, data_2)  # type: ignore
