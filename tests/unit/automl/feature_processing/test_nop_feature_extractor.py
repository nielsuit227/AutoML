#  Copyright (c) 2022 by Amplo.

import json

import pytest

from amplo.automl.feature_processing.nop_feature_extractor import NopFeatureExtractor


class TestNopFeatureExtractor:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_all(self, mode, make_x_y):
        x, y = make_x_y
        fe = NopFeatureExtractor()

        # Test fit and fit_transform
        out1 = fe.fit_transform(x, y)
        out2 = fe.transform(x)
        assert all(out1 == out2), "`fit_transform` and `transform` don't match."

        # Test features_
        assert set(fe.features_) == set(x), "Not all / too many features accepted."

        # Test settings
        new_fe = NopFeatureExtractor().load_settings(fe.get_settings())
        new_out = new_fe.transform(x)
        assert set(new_fe.features_) == set(x), "Features not correctly restored."
        assert all(out1 == new_out), "Transformation not correct."

        # Test JSON serializable
        settings = json.loads(json.dumps(fe.get_settings()))
        new_fe = NopFeatureExtractor().load_settings(settings)
        assert fe.get_settings() == new_fe.get_settings()
        assert all(fe.transform(x) == new_fe.transform(x))
