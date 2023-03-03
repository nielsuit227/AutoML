from polars.testing import assert_frame_equal

from amplo.automl.feature_processing.temporal_feature_extractor import (
    TemporalFeatureExtractor,
)
from amplo.utils.data import pandas_to_polars


class TestTemporalFeatureExtractor:
    def test_set_features(self):
        extractor = TemporalFeatureExtractor(target="target", mode="classification")
        extractor.set_features(["b__pool=cid_ce", "a__wav__gaus4__1.6__pool=abs_max"])
        assert extractor.raw_aggregator.features_ == ["b__pool=cid_ce"]
        assert extractor.wavelet_extractor.features_ == ["a__wav__gaus4__1.6"]
        assert extractor.wavelet_aggregator.features_ == [
            "a__wav__gaus4__1.6__pool=abs_max"
        ]

    def test_fit_transform(self, multiindex_data):
        pl_data, index_renaming = pandas_to_polars(multiindex_data)
        index_cols = list(index_renaming)

        extractor = TemporalFeatureExtractor(target="target", mode="classification")
        out1 = extractor.fit_transform(pl_data, index_cols)
        out2 = extractor.transform(pl_data, index_cols)
        assert_frame_equal(out1, out2)
