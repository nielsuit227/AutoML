#  Copyright (c) 2022 by Amplo.

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from amplo.automl.feature_processing._base import assert_double_index, check_data
from amplo.automl.feature_processing.nop_feature_extractor import NopFeatureExtractor
from amplo.utils.data import pandas_to_polars


@pytest.mark.usefixtures("random_number_generator")
class TestBaseFeatureExtractor:
    rng: np.random.Generator

    def test_setting_features(self):
        fe = NopFeatureExtractor(mode="classification")

        # Test set_features
        initial_features = [f"feat_{i}" for i in range(10)]
        fe.set_features(initial_features)
        assert set(fe.features_) == set(initial_features)

        # Test add_features
        added_features = [f"feat_{i}" for i in range(10, 20)]
        fe.add_features(added_features)
        fe.add_features(added_features)  # this should have no effect
        assert set(fe.features_) == set(initial_features).union(added_features)

    def test_scoring(self):
        # Init
        fe = NopFeatureExtractor(mode="classification")
        size = 100
        y = pl.Series([0, 1] * (size // 2))
        x = pl.DataFrame({"1to1": y, "random": self.rng.geometric(0.5, size)})

        # Test calc_feature_score
        scores: dict[str, float] = {}
        for col in x.columns:
            scores[col] = fe.calc_feature_score(x[col], y)
        assert scores["1to1"] >= -1e-9
        assert scores["random"] < -0.9

        # Test _init_feature_baseline_scores
        fe.initialize_baseline(x, y)
        assert fe._baseline_score == max(scores.values())

        # Test _update_feature_baseline_scores
        fe.update_baseline(1.0)
        assert fe._baseline_score == 1.0

    def test_accept_feature(self):
        fe = NopFeatureExtractor(mode="classification")
        fe.update_baseline(0.5)
        assert fe.accept_feature(np.array([0.6, 0.6]))
        assert fe.accept_feature(np.array([0.6, 0.1]))
        assert fe.accept_feature(np.array([0.1, 0.6]))
        assert fe.accept_feature(np.array([0.1, 0.1]))
        fe.set_features([f"a{i}" for i in range(50)])
        assert not fe.accept_feature(np.array([0.1, 0.1]))

    def test_select_scores(self):
        fe = NopFeatureExtractor(mode="classification")
        fe.set_features([f"a{i}" for i in range(50)])
        fe.update_baseline(0.5)
        scores = {
            "good_1": 0.6,
            "good_2": 0.55,
            "good_3": 0.87,
            "bad": 0.1,
        }
        accepted_scores = fe.select_scores(scores)
        assert set(accepted_scores) == {
            f"good_{i+1}" for i in range(3)
        }, "Accepted bad."


def test_check_data():
    with pytest.raises(ValueError):
        check_data(pl.DataFrame({"a__b": [0, 1]}))
    with pytest.raises(ValueError):
        check_data(pl.DataFrame({"a": [0, 1e15]}))
    with pytest.raises(ValueError):
        check_data(pl.DataFrame({"a": [0, None]}))
    with pytest.raises(ValueError):
        check_data(pl.DataFrame({"a": [0, np.nan]}))
    with pytest.raises(ValueError):
        check_data(pl.DataFrame({"a": ["a", "b"]}))


def test_assert_double_index(multiindex_data, classification_data):
    data, index_renaming = pandas_to_polars(multiindex_data)
    index_cols = list(index_renaming)

    # Double-index
    data, index_cols, was_double = assert_double_index(data, index_cols)
    assert was_double

    # Single-index
    data.drop_in_place(index_cols.pop(0))

    with pytest.raises(ValueError):
        data, index_cols, was_double = assert_double_index(data, index_cols)

    new_data, new_index_cols, was_double = assert_double_index(
        data, index_cols, allow_single=True
    )
    assert not was_double
    assert_frame_equal(data.drop(index_cols), new_data.drop(new_index_cols))
