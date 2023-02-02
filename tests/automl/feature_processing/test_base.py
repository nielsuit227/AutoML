#  Copyright (c) 2022 by Amplo.

import numpy as np
import pandas as pd
import pytest

from amplo.automl.feature_processing._base import assert_multi_index, check_data
from amplo.automl.feature_processing.nop_feature_extractor import NopFeatureExtractor


@pytest.mark.usefixtures("random_number_generator")
class TestBaseFeatureExtractor:
    rng: np.random.Generator

    def test_setting_features(self):
        fe = NopFeatureExtractor()

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

    def test_scoring(self):
        # Init
        mode = "classification"
        fe = NopFeatureExtractor(mode=mode)
        size = 100
        y = pd.Series([0] * size)
        y.iloc[::2] = 1
        x = pd.DataFrame({"1to1": y, "random": self.rng.geometric(0.5, size)})

        # Test calc_feature_score
        scores = x.apply(fe.calc_feature_score, y=y, axis=0)
        assert scores["1to1"] >= -1e-9
        assert scores["random"] < -0.9

        # Test _init_feature_baseline_scores
        fe.initialize_baseline(x, y)
        assert fe._baseline_score == scores.max()

        # Test _update_feature_baseline_scores
        fe.update_baseline(1.0)
        assert fe._baseline_score == 1.0

    def test_accept_feature(self):
        fe = NopFeatureExtractor()
        fe.update_baseline(0.5)
        assert fe.accept_feature(np.array([0.6, 0.6]))
        assert fe.accept_feature(np.array([0.6, 0.1]))
        assert fe.accept_feature(np.array([0.1, 0.6]))
        assert fe.accept_feature(np.array([0.1, 0.1]))
        fe.set_features([f"a{i}" for i in range(50)])
        assert not fe.accept_feature(np.array([0.1, 0.1]))

    def test_select_scores(self):
        fe = NopFeatureExtractor()
        fe.set_features([f"a{i}" for i in range(50)])
        fe.update_baseline(0.5)
        scores = pd.Series(
            {
                "good_1": 0.6,
                "good_2": 0.55,
                "good_3": 0.87,
                "bad": 0.1,
            }
        )
        accepted_scores = fe.select_scores(scores)
        assert set(accepted_scores) == set(scores.drop("bad", axis=0)), "Accepted bad."


def test_check_data():
    with pytest.raises(ValueError):
        check_data(pd.DataFrame({"a__b": [0, 1]}))
    with pytest.raises(ValueError):
        df = pd.DataFrame({"a": [0, 1]})
        check_data(pd.concat([df, df], axis=1))
    with pytest.raises(ValueError):
        check_data(pd.DataFrame({"a": [0, 1e15]}))
    with pytest.raises(ValueError):
        check_data(pd.DataFrame({"a": [0, np.nan]}))
    with pytest.raises(ValueError):
        check_data(pd.DataFrame({"a": ["a", "b"]}))


def test_multiindex_assert(multiindex_data, classification_data):
    with pytest.raises(ValueError):
        assert_multi_index(classification_data)
    _, was_multi = assert_multi_index(classification_data, allow_single_index=True)
    assert not was_multi
    dft, was_multi = assert_multi_index(multiindex_data)
    assert dft.equals(multiindex_data)
    assert was_multi
