#  Copyright (c) 2022 by Amplo.

import pytest

import amplo
from amplo.automl.feature_processing.nop_feature_extractor import NopFeatureExtractor


class TestNopFeatureExtractor:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_all(self, mode, data):
        fe = NopFeatureExtractor(target="target")

        # Test fit and fit_transform
        out1 = fe.fit_transform(data)
        out2 = fe.transform(data)
        assert all(out1 == out2), "`fit_transform` and `transform` don't match."

        # Test features_
        features = set(data) - {"target"}
        assert set(fe.features_) == features, "Not all / too many features accepted."

        # Test JSON serializable
        new_fe = amplo.loads(amplo.dumps(fe))
        assert all(fe.transform(data) == new_fe.transform(data))
