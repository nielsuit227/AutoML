#  Copyright (c) 2022 by Amplo.

import json

import numpy as np
import pandas as pd
import pytest

from amplo.automl.feature_processing.static_feature_extractor import (
    StaticFeatureExtractor,
)


@pytest.mark.usefixtures("make_rng")
class TestStaticFeatureExtractor:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_mode_and_settings(self, mode, make_x_y):
        x, y = make_x_y
        x = x.iloc[:, :5]  # for speed up
        fe = StaticFeatureExtractor(mode=mode)

        # Test output
        out1 = fe.fit_transform(x, y)
        out2 = fe.transform(x)
        assert set(out1) == set(fe.features_), "`features_` doesn't match output."
        assert all(out1 == out2), "`fit_transform` and `transform` don't match."

        # Test settings
        new_fe = StaticFeatureExtractor().load_settings(fe.get_settings())
        assert fe.get_settings() == new_fe.get_settings()
        new_out = new_fe.transform(x)
        assert all(out1 == new_out), "FE loaded from settings has invalid output."
        assert set(fe.features_) == set(
            new_fe.features_
        ), "FE from settings has erroneous `features_`."

        # Test JSON serializable
        settings = json.loads(json.dumps(fe.get_settings()))
        new_fe = StaticFeatureExtractor().load_settings(settings)
        assert fe.get_settings() == new_fe.get_settings()
        assert all(fe.transform(x) == new_fe.transform(x))

    @pytest.mark.parametrize("mode", ["classification"])
    def test_raw_features(self, mode, make_x_y):
        x, y = make_x_y

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        fe._fit_transform_raw_features(x, y)
        fe._is_fitted = True
        assert set(fe.features_) == set(fe.raw_features_)
        assert set(fe.features_) == set(x), "All columns should be accepted."

        # Test settings and transformation
        new_fe = StaticFeatureExtractor().load_settings(fe.get_settings())
        out = new_fe.transform(x)
        assert set(fe.features_) == set(out), "Expected columns don't match."

    @pytest.mark.parametrize("mode", ["regression"])
    def test_cross_features(self, mode):
        size = 100
        y = pd.Series(np.linspace(2, 100, size))
        random = pd.Series(self.rng.geometric(0.1, size))
        x = pd.DataFrame({"a": y / random, "b": y * random, "random": random})

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        fe._set_validation_model()
        fe._init_feature_baseline_scores(x, y)
        fe._fit_transform_cross_features(x, y)
        fe._is_fitted = True
        assert set(fe.features_) == set(fe.cross_features_)
        assert "a__mul__random" in fe.features_, "Multiplicative feature not found."
        assert "b__div__random" in fe.features_, "Division feature not found."

        # Test settings and transformation
        new_fe = StaticFeatureExtractor().load_settings(fe.get_settings())
        out = new_fe.transform(x)
        assert set(fe.features_) == set(out), "Expected columns don't match."

    @pytest.mark.parametrize("mode", ["classification"])
    def test_k_means_features(self, mode):
        size = 100
        y = pd.Series([0] * (size // 2) + [1] * (size // 2))
        blob1 = self.rng.normal(0, 1, size // 2)
        blob2 = self.rng.normal(100, 1, size // 2)
        random = pd.Series(self.rng.geometric(0.1, size))
        x = pd.DataFrame({"blobs": [*blob1, *blob2], "random": random})

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        fe._set_validation_model()
        fe._init_feature_baseline_scores(x, y)
        fe._fit_transform_k_means_features(x, y)
        fe._is_fitted = True
        assert set(fe.features_) == set(fe.k_means_features_)
        assert {"dist__0_2", "dist__1_2"}.issubset(
            fe.features_
        ), "K-Means features not found."

        # Test settings and transformation
        new_fe = StaticFeatureExtractor().load_settings(fe.get_settings())
        out = new_fe.transform(x)
        assert set(fe.features_) == set(out), "Expected columns don't match."

    @pytest.mark.parametrize("mode", ["regression"])
    def test_trigo_features(self, mode, make_x_y):
        size = 100
        y = pd.Series(self.rng.uniform(-1, 1, size=size))
        random = pd.Series(self.rng.geometric(0.1, size))
        x = pd.DataFrame(
            {"sinus": np.arcsin(y), "cosine": np.arccos(y), "random": random}
        )

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        fe._set_validation_model()
        fe._init_feature_baseline_scores(x, y)
        fe._fit_transform_trigo_features(x, y)
        fe._is_fitted = True
        assert set(fe.features_) == set(fe.trigo_features_)
        assert "sin__sinus" in fe.features_, "Sinus feature not found."
        assert "cos__cosine" in fe.features_, "Cosine feature not found."

        # Test settings and transformation
        new_fe = StaticFeatureExtractor().load_settings(fe.get_settings())
        out = new_fe.transform(x)
        assert set(fe.features_) == set(out), "Expected columns don't match."

    @pytest.mark.parametrize("mode", ["regression"])
    def test_inverse_features(self, mode):
        size = 100
        y = pd.Series(self.rng.uniform(-1, 1, size=size))
        random = pd.Series(self.rng.geometric(0.1, size))
        x = pd.DataFrame({"inversed": (1.0 / y), "random": random})

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        x = fe._check_x(x)  # sanitize
        fe._set_validation_model()
        fe._init_feature_baseline_scores(x, y)
        fe._fit_transform_inverse_features(x, y)
        fe._is_fitted = True
        assert set(fe.features_) == set(fe.inverse_features_)
        assert "inv__inversed" in fe.features_, "Inverse feature not found."

        # Test settings and transformation
        new_fe = StaticFeatureExtractor().load_settings(fe.get_settings())
        out = new_fe.transform(x)
        assert set(fe.features_) == set(out), "Expected columns don't match."
