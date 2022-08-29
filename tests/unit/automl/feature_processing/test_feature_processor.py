#  Copyright (c) 2022 by Amplo.

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from amplo.automl.feature_processing.feature_processor import (
    FeatureProcessor,
    find_collinear_columns,
)
from amplo.automl.feature_processing.nop_feature_extractor import NopFeatureExtractor
from amplo.automl.feature_processing.temporal_feature_extractor import (
    TemporalFeatureExtractor,
)


@pytest.mark.usefixtures("make_rng")
class TestFunctions:
    def test_find_collinear_columns(self):
        col = np.linspace(-4, 4, 100)
        x = pd.DataFrame(
            {"a": col, "b": col, "c": col + self.rng.normal(scale=0.1, size=100)}
        )
        collinear_columns = find_collinear_columns(x, information_threshold=0.99)
        assert set(collinear_columns) == {"b", "c"}, "Collinear columns not found"


@pytest.mark.usefixtures("make_rng")
class TestFeatureProcessor:
    @pytest.mark.parametrize("mode", ["classification", "multiclass", "regression"])
    @pytest.mark.parametrize("extraction", ["nop", "static", "temporal"])
    def test_transform(self, mode, make_x_y, extraction):
        x, y = make_x_y
        x = x.iloc[:, :5]  # for speed up
        kwargs = {"mode": mode, "analyse_feature_sets": None}
        if extraction == "nop":
            fp = FeatureProcessor(extract_features=False, **kwargs)
        elif extraction == "static":
            fp = FeatureProcessor(extract_features=True, is_temporal=False, **kwargs)
        elif extraction == "temporal":
            index = pd.MultiIndex.from_product([[0, 1], range(len(x) // 2)])
            x.index = index
            y.index = index
            kwargs = {**kwargs, "timeout": 1}
            fp = FeatureProcessor(extract_features=True, is_temporal=True, **kwargs)
        else:
            raise ValueError("Invalid extraction mode.")

        # Test equivalence of transform and fit_transform
        out1 = fp.fit_transform(x, y)
        out2 = fp.transform(x)
        assert all(out1 == out2)

        # Test transform_target
        if isinstance(fp.feature_extractor, TemporalFeatureExtractor):
            y_transformed = fp.feature_extractor._fit_data_to_window_size(y)
            y_transformed = fp.feature_extractor._pool_target(y_transformed)
        else:
            y_transformed = y
        assert all(y_transformed == fp.transform_target(y))

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    def test_settings(self, mode, make_x_y):
        x, y = make_x_y
        fp = FeatureProcessor(
            mode=mode, extract_features=False, analyse_feature_sets=None
        )
        fp.fit(x, y)

        # Test load settings directly
        new_fp = FeatureProcessor().load_settings(fp.get_settings())
        assert fp.get_settings() == new_fp.get_settings()
        assert all(fp.transform(x) == new_fp.transform(x))

        # Test JSON serializable
        settings = json.loads(json.dumps(fp.get_settings()))
        new_fp = FeatureProcessor().load_settings(settings)
        assert fp.get_settings() == new_fp.get_settings()
        assert all(fp.transform(x) == new_fp.transform(x))

    @pytest.mark.parametrize("is_temporal", [True, False])
    def test_check_is_temporal_attribute(self, is_temporal):
        size = 100
        if is_temporal:
            index = pd.MultiIndex.from_product([[0], range(size)])
        else:
            index = pd.RangeIndex(size)
        data = pd.DataFrame(index=index)

        fp = FeatureProcessor()
        assert fp.is_temporal is None, "Attribute shouldn't be set yet."
        fp._check_is_temporal_attribute(data)
        assert fp.is_temporal == is_temporal, "Attribute is set wrong."

    def test_find_columns_of_interest(self):
        x = pd.DataFrame(
            {
                "A": [1, 4, 1],
                "B": [10, 9, 1],
                "C": [datetime.now()] * 3,
                "D": [39, 0, 0],
                "collinear": [1, 4, 1],
            }
        )
        fp = FeatureProcessor(extract_features=False)
        fp._find_columns_of_interest(x)
        fp.feature_extractor = NopFeatureExtractor()
        fp.features_ = ["A__mul__B", "inv__A", "...__C__..."]
        assert set(fp.numeric_cols_) == set(list("ABD"))
        assert set(fp.datetime_cols_) == {"C"}
        assert set(fp.collinear_cols_) == {"collinear"}
        assert set(fp.get_required_columns()) == set(list("ABC"))

    @pytest.mark.parametrize("mode", ["classification", "regression"])
    @pytest.mark.parametrize("analyse_fs", [None, "auto", "all", "gini", "shap"])
    def test_feature_sets(self, mode, analyse_fs):
        size = 100
        y = pd.Series(self.rng.choice([0, 1], size))
        x = pd.DataFrame({"y": y, "zeros": np.zeros(size)})
        fp = FeatureProcessor(
            mode=mode,
            extract_features=False,
            analyse_feature_sets=analyse_fs,
            selection_cutoff=0.99,
            selection_increment=0.01,
        ).fit(x, y)

        if analyse_fs is None:
            assert set(fp.feature_sets_) == {"take_all"}
            assert set(fp.feature_sets_["take_all"]) == {"y", "zeros"}
        else:
            # Check validity of feature sets
            for name, values in fp.feature_sets_.items():
                assert set(values) == {"y"}, "Invalid feature set."
                x_out = fp.transform(x, feature_set=name)
                assert all(x_out == x[["y"]]), "Erroneous feature set transformation."

            # Check that desired feature sets are present
            expected_fs = set()
            if analyse_fs in ("auto", "all", "gini"):
                expected_fs = {*expected_fs, "rf_increment", "rf_threshold"}
            if analyse_fs in ("auto", "all", "shap"):
                expected_fs = {*expected_fs, "shap_increment", "shap_threshold"}
            assert set(fp.feature_sets_) == expected_fs

        # Additional test for shap features:
        # When data size is >= 5000 samples, don't calculate shap features.
        if analyse_fs == "auto":
            size = 50_000
            y = pd.Series(self.rng.choice([0, 1], size))
            x = pd.DataFrame({"y": y, "zeros": np.zeros(size)})
            fp._analyse_feature_sets(x, y)
            assert "shap_increment" not in fp.feature_sets_
            assert "shap_threshold" not in fp.feature_sets_

    def test_impute_missing_columns(self):
        # Make data
        size = 100
        numeric = pd.DataFrame(
            self.rng.normal(size=(size, 5)), columns=[f"numeric_{i}" for i in range(5)]
        )
        date = pd.DataFrame(np.array([datetime.now()] * size), columns=["datetime_0"])
        collinear = numeric.rename(
            columns={col: f"collinear_{i}" for i, col in enumerate(numeric)}
        )

        # Init feature processor
        fp = FeatureProcessor(mode="classification")
        fp._is_fitted = True
        fp.feature_extractor = NopFeatureExtractor()
        fp.feature_extractor._is_fitted = True
        fp.numeric_cols_ = list(numeric.columns)
        fp.datetime_cols_ = list(date.columns)
        fp.collinear_cols_ = list(collinear.columns)
        fp.features_ = [*fp.numeric_cols_, *fp.datetime_cols_]

        # Check no imputation
        x = pd.concat([numeric, date], axis=1)
        imputed = fp._impute_missing_columns(x)
        assert set(x) == set(imputed), "Not all / too many columns are present."
        assert all(x == imputed), "Invalid imputation of values."

        # Check numeric imputation
        x = pd.concat([numeric, date], axis=1)
        imputed = fp._impute_missing_columns(x.drop("numeric_0", axis=1))
        x.loc[:, "numeric_0"] = 0
        assert set(x) == set(imputed), "Not all / too many columns are present."
        assert all(x == imputed[x.columns]), "Erroneous numeric value imputation."
