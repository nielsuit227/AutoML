import numpy as np
import polars as pl
import pytest

from amplo.automl.feature_processing.feature_processing import (
    FeatureProcessor,
    find_collinear_columns,
    get_required_columns,
    translate_features,
)
from amplo.automl.feature_processing.nop_feature_extractor import NopFeatureExtractor
from amplo.automl.feature_processing.static_feature_extractor import (
    StaticFeatureExtractor,
)
from amplo.automl.feature_processing.temporal_feature_extractor import (
    TemporalFeatureExtractor,
)


class TestFeatureProcessor:
    def test_find_collinear_columns(self):
        # Test clean data
        a = np.random.normal(0, 1, 100)
        b = np.random.normal(0, 1, 100)
        noise = np.random.normal(0, 0.1, 100)
        df = pl.DataFrame({"a": a, "a_collinear": a + noise, "b": b})
        assert find_collinear_columns(df, 0.0) == ["a_collinear", "b"]
        assert find_collinear_columns(df, 0.9) == ["a_collinear"]
        assert find_collinear_columns(df, 0.999) == []

        # Test data with NaN values
        a_nan = a.copy()
        a_nan[::3] = None
        df_nan = pl.DataFrame({"a": a_nan, "a_collinear": a_nan + noise, "b": b})
        assert df_nan["a"].is_nan().sum()  # in polars: NaN != null
        assert find_collinear_columns(df_nan, 0.0) == ["a_collinear", "b"]
        assert find_collinear_columns(df_nan, 0.9) == ["a_collinear"]
        assert find_collinear_columns(df_nan, 0.999) == []

        # Test data with null values
        df_null = df_nan.with_columns(pl.all().fill_nan(None))
        assert df_null["a"].is_null().sum()  # in polars: NaN != null
        assert find_collinear_columns(df_null, 0.0) == ["a_collinear", "b"]
        assert find_collinear_columns(df_null, 0.9) == ["a_collinear"]
        assert find_collinear_columns(df_null, 0.999) == []

    def test_translate_features(self):
        features = {
            "a__mul__b": ["a", "b"],
            "sin__c": ["c"],
            "d__pool=abs_max": ["d"],
            "e__pool=cid_ce": ["e"],
            "g__wav__gaus4__1.4__pool=abs_energy": ["g"],
            "h": ["h"],
        }
        features = translate_features(list(features.keys()))

    def get_required_columns(self):
        features = [
            "a__mul__b",
            "sin__c",
            "d__pool=abs_max",
            "e__pool=cid_ce",
            "g__wav__gaus4__1.4__pool=abs_energy",
            "h",
        ]
        assert {"a", "b", "c", "d", "e", "g", "h"} == set(
            get_required_columns(features)
        )

    def test_set_extractor(self):
        processor = FeatureProcessor(extract_features=False, mode="classification")
        processor._set_feature_extractor(["index"])
        assert isinstance(processor.feature_extractor, NopFeatureExtractor)

        processor = FeatureProcessor(mode="classification")
        processor._set_feature_extractor(["index"])
        assert isinstance(processor.feature_extractor, StaticFeatureExtractor)

        processor = FeatureProcessor(is_temporal=False, mode="classification")
        processor._set_feature_extractor(["index", "index2"])
        assert isinstance(processor.feature_extractor, StaticFeatureExtractor)

        processor = FeatureProcessor(mode="classification")
        processor._set_feature_extractor(["index", "index2"])
        assert isinstance(processor.feature_extractor, TemporalFeatureExtractor)

        processor = FeatureProcessor(is_temporal=True, mode="classification")
        processor._set_feature_extractor(["index"])
        assert isinstance(processor.feature_extractor, TemporalFeatureExtractor)

    def test_temporal_feature_pruning(self, multiindex_data):
        processor = FeatureProcessor(target="target", mode="classification")
        processor._set_feature_extractor(multiindex_data)
        processor.is_fitted_ = True
        processor.feature_selector.is_fitted_ = True
        processor.feature_extractor.is_fitted_ = True
        processor.feature_selector.feature_sets_ = {
            "rf_increment": ["a__wav__gaus4__1.6__pool=abs_max", "b__pool=cid_ce"]
        }
        processor.set_feature_set("rf_increment")

        assert processor.feature_set_ == "rf_increment"
        assert processor.features_ == [
            "a__wav__gaus4__1.6__pool=abs_max",
            "b__pool=cid_ce",
        ]

    def test_transform_with_missing_columns(self, classification_data):
        processor = FeatureProcessor(target="target", mode="classification")
        processor.fit_transform(classification_data)

        # Drop a column and process -- should still work, but warn
        with pytest.warns():
            processor.transform(
                classification_data.drop(classification_data.keys()[0], axis=1)
            )
