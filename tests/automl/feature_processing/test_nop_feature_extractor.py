#  Copyright (c) 2022 by Amplo.

import pytest
from polars.testing import assert_frame_equal

import amplo
from amplo.automl.feature_processing.nop_feature_extractor import NopFeatureExtractor
from amplo.utils.data import pandas_to_polars


class TestNopFeatureExtractor:
    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_all(self, mode, data):
        pl_data, index_renaming = pandas_to_polars(data)
        index_cols = list(index_renaming)
        fe = NopFeatureExtractor(target="target", mode=mode)

        # Test fit and fit_transform
        out1 = fe.fit_transform(pl_data, index_cols)
        out2 = fe.transform(pl_data, index_cols)
        assert_frame_equal(out1, out2)

        # Test features_
        features = set(data) - {"target"}
        assert set(fe.features_) == features, "Not all / too many features accepted."

        # Test JSON serializable
        new_fe: NopFeatureExtractor = amplo.loads(amplo.dumps(fe))
        assert_frame_equal(
            fe.transform(pl_data, index_cols), new_fe.transform(pl_data, index_cols)
        )
