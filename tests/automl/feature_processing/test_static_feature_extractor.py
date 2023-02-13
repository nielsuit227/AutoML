#  Copyright (c) 2022 by Amplo.

import numpy as np
import pandas as pd
import pytest
from numpy.random import Generator

import amplo
from amplo.automl.feature_processing.static_feature_extractor import (
    StaticFeatureExtractor,
)


@pytest.mark.usefixtures("random_number_generator")
class TestStaticFeatureExtractor:
    rng: Generator

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_mode_and_settings(self, mode, x_y):
        x, y = x_y
        x = x.iloc[:, :5]  # for speed up
        data = pd.DataFrame(x).copy()
        data["target"] = y
        fe = StaticFeatureExtractor(target="target", mode=mode)

        # Test output
        out1 = fe.fit_transform(data)
        out2 = fe.transform(data)
        assert set(out1) == set(
            fe.features_ + ["target"]
        ), "`features_` doesn't match output."
        assert out1.equals(out2), "`fit_transform` and `transform` don't match."

        # Test settings
        new_fe = amplo.loads(amplo.dumps(fe))
        assert out1.equals(
            new_fe.transform(data)
        ), "FE loaded from settings has invalid output."
        assert set(fe.features_) == set(
            new_fe.features_
        ), "FE from settings has erroneous `features_`."

    def test_raw_features(self):
        mode = "regression"
        x = pd.DataFrame({"a": np.linspace(0, 100, 100)})

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        fe._fit_transform_raw_features(x)
        fe.is_fitted_ = True
        assert set(fe.features_) == set(fe.raw_features_)
        assert set(fe.features_) == set(x), "All columns should be accepted."

        # Test settings and transformation
        new_fe = amplo.loads(amplo.dumps(fe))
        out = new_fe.transform(x)
        assert set(fe.features_) == set(out), "Expected columns don't match."

    def test_cross_features(self):
        mode = "regression"
        size = 100
        y = pd.Series(np.linspace(2, 100, size))
        random = pd.Series(self.rng.geometric(0.1, size))
        x = pd.DataFrame({"a": y / random, "b": y * random, "random": random})

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        fe.initialize_baseline(x, y)
        fe._fit_transform_cross_features(x, y)
        fe.is_fitted_ = True
        assert set(fe.features_) == set(fe.cross_features_)
        assert "a__mul__random" in fe.features_, "Multiplicative feature not found."
        assert "b__div__random" in fe.features_, "Division feature not found."

        # Test settings and transformation
        new_fe = amplo.loads(amplo.dumps(fe))
        out = new_fe.transform(x)
        assert set(fe.features_) == set(out), "Expected columns don't match."

    def test_trigo_features(self):
        mode = "regression"
        size = 100
        y = pd.Series(self.rng.uniform(-1, 1, size=size))
        random = pd.Series(self.rng.geometric(0.1, size))
        x = pd.DataFrame(
            {"sinus": np.arcsin(y), "cosine": np.arccos(y), "random": random}
        )

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        fe.initialize_baseline(x, y)
        fe._fit_transform_trigo_features(x, y)
        fe.is_fitted_ = True
        assert set(fe.features_) == set(fe.trigo_features_)
        assert "sin__sinus" in fe.features_, "Sinus feature not found."
        assert "cos__cosine" in fe.features_, "Cosine feature not found."

        # Test settings and transformation
        new_fe = amplo.loads(amplo.dumps(fe))
        out = new_fe.transform(x)
        assert set(fe.features_) == set(out), "Expected columns don't match."

    def test_inverse_features(self):
        mode = "regression"
        size = 100
        y = pd.Series(self.rng.uniform(-1, 1, size=size))
        random = pd.Series(self.rng.geometric(0.1, size))
        x = pd.DataFrame({"inversed": (1.0 / y), "random": random})

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        fe.initialize_baseline(x, y)
        fe._fit_transform_inverse_features(x, y)
        fe.is_fitted_ = True
        assert set(fe.features_) == set(fe.inverse_features_)
        assert "inv__inversed" in fe.features_, "Inverse feature not found."

        # Test settings and transformation
        new_fe = amplo.loads(amplo.dumps(fe))
        out = new_fe.transform(x)
        assert set(fe.features_) == set(out), "Expected columns don't match."
