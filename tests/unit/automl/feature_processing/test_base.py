#  Copyright (c) 2022 by Amplo.

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from amplo.automl.feature_processing._base import (
    BaseFeatureExtractor,
    BaseFeatureProcessor,
    sanitize_dataframe,
    sanitize_series,
)
from tests import make_x_y


@pytest.fixture(scope="class")
def make_good_and_corrupt_classification():
    # Make corrupted data
    x_corrupt, y_corrupt = make_x_y(mode="classification")
    x_corrupt.iloc[0, 0] = np.nan
    y_corrupt.iloc[0] = np.nan
    x_corrupt.columns = list(range(len(x_corrupt.columns)))
    y_corrupt.name = None

    # Make sanitized data
    x_good = sanitize_dataframe(x_corrupt)
    x_good.columns = list(map(str, x_good.columns))
    y_good = sanitize_series(y_corrupt)
    y_good.name = "target"

    yield x_good, x_corrupt, y_good, y_corrupt


class TestFunctions:
    def test_sanitize_series_and_dataframe(self):
        frame = pd.DataFrame(
            {
                "non_numeric": list("abcd"),
                "floats": [123, 1e12 + 1, -1e12 - 1, np.nan],
                "ints": [123, 1e12 + 1, -1e12 - 1, 0],
            }
        )
        frame["ints"] = frame["ints"].astype(int)
        sanitized_frame = pd.DataFrame(
            {
                "non_numeric": list("abcd"),
                "floats": [123, 1e12, -1e12, 0],
                "ints": [123, 2147483647, -2147483648, 0],
            }
        )
        sanitized_frame["floats"] = sanitized_frame["floats"].astype(np.float32)
        sanitized_frame["ints"] = sanitized_frame["ints"].astype(np.int32)

        # Check sanitize_series
        for col in frame:
            assert all(sanitized_frame[col] == sanitize_series(frame[col]))

        # Check sanitize_dataframe
        assert all(sanitized_frame == sanitize_dataframe(frame))


class TestBaseFeatureProcessor:
    def test_mode(self):
        # Test notset
        fp = BaseFeatureProcessor(mode=None)
        assert fp.mode == "notset"
        fp = BaseFeatureProcessor(mode="notset")
        assert fp.mode == "notset"
        # Test classification
        fp = BaseFeatureProcessor(mode="classification")
        assert fp.mode == "classification"
        fp = BaseFeatureProcessor(mode="multiclass")
        assert fp.mode == "classification"
        # Test regression
        fp = BaseFeatureProcessor(mode="regression")
        assert fp.mode == "regression"
        # Test cases
        fp = BaseFeatureProcessor(mode="NOTSET")
        assert fp.mode == "notset"
        # Test error
        with pytest.raises(ValueError):
            # Should error when trying to set an unknown mode.
            BaseFeatureProcessor(mode="invalid_mode")

    def test_check_x(self, make_good_and_corrupt_classification):
        fp = BaseFeatureProcessor  # has not to be instantiated
        x_good, x_corrupt, _, _ = make_good_and_corrupt_classification

        # Check good input
        checked = fp._check_x(x_good)
        assert np.allclose(checked, x_good), "Good x data was handled unexpectedly."

        # Check corrupted input
        checked = fp._check_x(x_corrupt)
        assert np.allclose(checked, x_good), "Invalid data sanitization."
        assert all(isinstance(c, str) for c in checked.columns), "Non-string columns."

        # Check error when non-unique column names
        x_corrupt_columns = x_corrupt.columns
        x_corrupt.columns = [0, 0, *x_corrupt.columns[2:]]
        with pytest.raises(ValueError):
            # Should raise error when receiving non-unique column names.
            fp._check_x(x_corrupt)
        x_corrupt.columns = x_corrupt_columns  # reset since fixture has scope="class"

    def test_check_y(self, make_good_and_corrupt_classification):
        fp = BaseFeatureProcessor  # has not to be instantiated
        _, _, y_good, y_corrupt = make_good_and_corrupt_classification

        # Check good input
        checked = fp._check_y(y_good)
        assert np.allclose(checked, y_good), "Good y data was handled unexpectedly."

        # Check corrupted input
        checked = fp._check_y(y_corrupt)
        assert np.allclose(checked, y_good), "Invalid data sanitization."
        assert isinstance(checked.name, str), "Non-string name."

    def test_check_x_y(self, make_good_and_corrupt_classification):
        fp = BaseFeatureProcessor  # has not to be instantiated
        x, _, y, _ = make_good_and_corrupt_classification

        # Check integrity
        y_short = y.iloc[:-2]
        with pytest.raises(ValueError):
            # Should raise error when receiving x and y of unequal length.
            fp._check_x_y(x, y_short)

        # Test warning for non-matching index
        y_bad_index = deepcopy(y)
        y_bad_index.index = pd.RangeIndex(1, len(y) + 1)
        with pytest.warns(UserWarning):
            fp._check_x_y(x, y_bad_index)


@pytest.mark.usefixtures("make_rng")
class TestBaseFeatureExtractor:
    def test_setting_features(self):
        fe = BaseFeatureExtractor()
        # Test set_features
        initial_features = [f"feat_{i}" for i in range(10)]
        fe.set_features(initial_features)
        assert set(fe.features_) == set(initial_features)
        # Test add_features
        added_features = [f"feat_{i}" for i in range(10, 20)]
        fe.add_features(added_features)
        fe.add_features(added_features)  # this should have no effect
        assert set(fe.features_) == set(initial_features).union(added_features)
        # Test remove_features
        fe.remove_features(added_features)
        assert set(fe.features_) == set(initial_features)
        with pytest.raises(ValueError):
            # Should raise error when trying to remove features that don't exist.
            fe.remove_features(["non_existing"])

    @pytest.mark.parametrize("mode", ["classification", "multiclass", "regression"])
    def test_scoring(self, mode):
        # Init
        fe = BaseFeatureExtractor(mode=mode)
        fe._set_validation_model()
        size = 100
        if mode == "classification":
            y = pd.Series([0] * size)
            y.iloc[::2] = 1
        elif mode == "multiclass":
            y = pd.Series([0] * size)
            y.iloc[::3] = 1
            y.iloc[1::3] = 2
        else:
            y = pd.Series(range(size))
        x = pd.DataFrame({"1to1": y, "random": self.rng.geometric(0.5, size)})

        # Test _calc_feature_scores
        scores = x.apply(fe._calc_feature_scores, y=y, axis=0)
        assert all(scores["1to1"] >= 0.9)
        assert all(scores["random"] < 0.9)

        # Test _init_feature_baseline_scores
        fe._init_feature_baseline_scores(x, y)
        assert all(np.array(fe._baseline_scores) == scores.max(1).values)

        # Test _update_feature_baseline_scores
        new_scores = pd.DataFrame({"new_score": [1.0] * len(scores)})
        fe._update_feature_baseline_scores(new_scores)
        assert all(np.array(fe._baseline_scores) == 1.0)

    def test_accept_feature(self):
        fe = BaseFeatureExtractor()
        fe._baseline_scores = [0.5, 0.5]
        assert fe.accept_feature([0.6, 0.6])
        assert fe.accept_feature([0.6, 0.1])
        assert fe.accept_feature([0.1, 0.6])
        assert not fe.accept_feature([0.1, 0.1])

    def test_select_scores(self):
        fe = BaseFeatureExtractor()
        fe._baseline_scores = [0.5, 0.5]
        scores = pd.DataFrame(
            {
                "good_1": [0.6, 0.6],
                "good_2": [0.6, 0.1],
                "good_3": [0.1, 0.6],
                "bad": [0.1, 0.1],
            }
        )
        accepted_scores = fe.select_scores(scores)
        assert set(accepted_scores) == set(scores.drop("bad", axis=1)), "Accepted bad."
