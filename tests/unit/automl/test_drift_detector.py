#  Copyright (c) 2022 by Amplo.

import json
import warnings

import numpy as np
import pytest
import scipy.stats

from amplo.automl import DriftDetector
from amplo.automl.drift_detection import DataDriftWarning
from amplo.utils.testing import (
    DummyDataSampler,
    make_cat_data,
    make_data,
    make_num_data,
)


class DummyPredictor(DummyDataSampler):
    def predict(self, data):
        return self.sample_data(len(data))


class TestDriftDetector:
    def test_distribution_fits(self):
        # Setup
        ref, cols = make_num_data(500)
        test = ref.iloc[np.random.permutation(len(ref))[:10]]
        drift = DriftDetector(**cols)
        drift.fit(ref)

        # Checks
        assert len(drift.check(test)) == 0, "Test data found inconsistent"
        assert len(drift.check(ref.max() + 1)) == len(
            ref.columns
        ), "Maxima not detected"
        assert len(drift.check(ref.min() - 1)) == len(
            ref.columns
        ), "Minima not detected"

    def test_categorical(self):
        df, cols = make_cat_data(10, list("abcd"))
        drift = DriftDetector(**cols)
        drift.fit(df)
        for col in df.columns:
            assert col in drift.bins, f"Column '{col}' rejected"
            assert drift.bins[col] == df[col].value_counts().to_dict()

    def test_add_bins(self):
        yp, _ = make_num_data(100, "randint::0::2")
        df, cols = make_cat_data(10, list("abcd"))
        drift = DriftDetector(**cols)
        drift.fit(df)

        # Test empty
        assert drift.add_bins({}, df)
        assert drift.add_output_bins((), yp)

        # Test actual adding
        new_bins = drift.add_bins(drift.bins, df)
        for col in df.columns:
            assert col in new_bins, f"Column '{col}' rejected"
            assert new_bins[col] == {
                key: 2 * value
                for key, value in df[col].value_counts().to_dict().items()
            }

    def test_storable(self):
        df, cols = make_data(10, cat_choices=list("abc"), num_dists="norm")
        drift = DriftDetector(**cols)
        drift.fit(df)
        json.dumps(drift.bins)
        json.dumps(drift.add_bins(drift.bins, df))
        pred = np.random.randint(0, 2, (100,))
        old = drift.add_output_bins((), pred)
        drift.add_output_bins(old, pred)

    def test_no_drift_warning(self):
        """
        Ensure that minor changes in data do not trigger warnings.
        """
        # Create dummy data
        data_1, cols_1 = make_data(
            500, num_dists=["uniform", "norm"], cat_choices=[list("abc"), list("abc")]
        )
        data_2, cols_2 = make_data(
            10,
            num_dists=[scipy.stats.gamma(1), scipy.stats.beta(1, 2)],
            cat_choices=[list("abc"), list("xyz")],
        )

        # Create dummy predictors
        dummy_model_1 = DummyPredictor()
        dummy_model_2 = DummyPredictor(scipy.stats.randint(0, 10))

        # Instantiate and fit drift detector
        drift = DriftDetector(**cols_1, n_bins=10)
        drift.fit(data_1)
        drift.fit_output(dummy_model_1, data_1)

        # Assert that no DataDriftWarning occurs when given same data and model
        with warnings.catch_warnings(record=True) as caught_warnings:
            # Check drift on input
            drift.check(data_1)
            # Check drift on output
            drift.check_output(dummy_model_1, data_1)
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
            drift.check_output(dummy_model_2, data_2)
