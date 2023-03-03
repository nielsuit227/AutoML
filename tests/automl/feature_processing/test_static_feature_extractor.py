#  Copyright (c) 2022 by Amplo.

import numpy as np
import polars as pl
import pytest
from numpy.random import Generator
from polars.testing import assert_frame_equal

import amplo
from amplo.automl.feature_processing.static_feature_extractor import (
    StaticFeatureExtractor,
)
from amplo.utils.data import pandas_to_polars


@pytest.mark.usefixtures("random_number_generator")
class TestStaticFeatureExtractor:
    rng: Generator

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_mode_and_settings(self, mode, data):
        pl_data, index_renaming = pandas_to_polars(data)
        index_cols = list(index_renaming)

        fe = StaticFeatureExtractor(target="target", mode=mode)

        # Test output
        out1 = fe.fit_transform(pl_data, index_cols)
        out2 = fe.transform(pl_data, index_cols)
        assert set(out1.columns) == {*index_cols, *fe.features_, "target"}
        assert_frame_equal(out1, out2)

        # Test settings
        new_fe: StaticFeatureExtractor = amplo.loads(amplo.dumps(fe))
        assert_frame_equal(out1, new_fe.transform(pl_data, index_cols))
        assert set(fe.features_) == set(
            new_fe.features_
        ), "FE from settings has erroneous `features_`."

    def test_raw_features(self):
        mode = "regression"
        x = pl.DataFrame({"a": np.linspace(0, 100, 100)})

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        fe._fit_transform_raw_features(x)
        fe.is_fitted_ = True
        assert set(fe.features_) == set(fe.raw_features_)
        assert set(fe.features_) == set(x.columns), "All columns should be accepted."

        # Test settings and transformation
        new_fe = amplo.loads(amplo.dumps(fe))
        out = new_fe._transform_raw_features(x)
        assert set(fe.features_) == set(out.columns), "Expected columns don't match."

    def test_cross_features(self):
        mode = "regression"
        size = 100
        y = pl.Series(np.linspace(2, 100, size))
        random = pl.Series(self.rng.geometric(0.1, size))
        x = pl.DataFrame({"a": y / random, "b": y * random, "random": random})

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
        out = new_fe._transform_cross_features(x)
        assert set(fe.features_) == set(out.columns), "Expected columns don't match."

    def test_trigo_features(self):
        mode = "regression"
        size = 100
        y = pl.Series(self.rng.uniform(-1, 1, size=size))
        random = pl.Series(self.rng.geometric(0.1, size))
        x = pl.DataFrame(
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
        out = new_fe._transform_trigo_features(x)
        assert set(fe.features_) == set(out.columns), "Expected columns don't match."

    def test_inverse_features(self):
        mode = "regression"
        size = 100
        y = pl.Series(self.rng.uniform(-1, 1, size=size))
        random = pl.Series(self.rng.geometric(0.1, size))
        x = pl.DataFrame({"inversed": (1.0 / y), "random": random})

        # Fit and check features
        fe = StaticFeatureExtractor(mode=mode)
        fe.initialize_baseline(x, y)
        fe._fit_transform_inverse_features(x, y)
        fe.is_fitted_ = True
        assert set(fe.features_) == set(fe.inverse_features_)
        assert "inv__inversed" in fe.features_, "Inverse feature not found."

        # Test settings and transformation
        new_fe = amplo.loads(amplo.dumps(fe))
        out = new_fe._transform_inverse_features(x)
        assert set(fe.features_) == set(out.columns), "Expected columns don't match."
