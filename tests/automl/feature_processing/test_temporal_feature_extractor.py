import numpy as np

from amplo.automl.feature_processing.temporal_feature_extractor import (
    TemporalFeatureExtractor,
)


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
        extractor = TemporalFeatureExtractor(target="target", mode="classification")
        dft = extractor.fit_transform(multiindex_data)
        assert np.allclose(dft, extractor.transform(multiindex_data))
