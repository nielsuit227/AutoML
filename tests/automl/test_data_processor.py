#  Copyright (c) 2022 by Amplo.

import datetime

import numpy as np
import pandas as pd
import polars as pl

import amplo
from amplo.automl import DataProcessor


class TestDataProcessor:
    def test_interpolation(self):
        dp = DataProcessor()
        assert dp.missing_values == "interpolate", "Unexpected default value"
        data = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan, 5], "b": [1, 2, 3, 4, 5]})
        cleaned = dp.fit_transform(data)
        assert cleaned["a"].astype(int).tolist() == [1, 2, 3, 4, 5]

    def test_type_detector_no_nan(self):
        dp = DataProcessor()
        data = pd.DataFrame(
            {
                "cat": np.random.choice(list("abc"), 100),
                "int": np.arange(100),
                "float": np.arange(0, 10, 0.1),
                "bool": np.random.choice([True, False], 100),
                "date": [
                    str(datetime.datetime.fromtimestamp(1673960000 + 100 * i))
                    for i in range(100)
                ],
            }
        )
        cleaned = dp.fit_transform(data)

        assert {
            "cat_a",
            "cat_b",
            "cat_c",
            "int",
            "date",
            "bool",
            "float",
        } == set(cleaned.columns), "Unexpected columns"
        assert len(dp.num_cols_) == 2
        assert set(dp.num_cols_) == {"float", "int"}
        assert len(dp.bool_cols_) == 1
        assert set(dp.bool_cols_) == {"bool"}
        assert len(dp.cat_cols_) == 1
        assert set(dp.cat_cols_) == {"cat"}

    def test_type_detector_with_nan(self):
        dp = DataProcessor()
        data = pd.DataFrame(
            {
                "cat": np.random.choice(list("abc"), 100),
                "int": np.arange(100),
                "float": np.arange(0, 10, 0.1),
                "bool": np.random.choice([True, False], 100),
                "date": [
                    str(datetime.datetime.fromtimestamp(1673960000 + 100 * i))
                    for i in range(100)
                ],
            }
        )
        # Randomly insert 10% NaN values
        ix = np.array(
            [(row, col) for row in range(data.shape[0]) for col in range(data.shape[1])]
        )
        for i in np.random.choice(np.arange(len(ix)), len(ix) // 10, replace=False):
            row, col = ix[i]
            data.iat[row, col] = np.nan

        cleaned = dp.fit_transform(data)

        assert {
            "cat_a",
            "cat_b",
            "cat_null",
            "cat_c",
            "int",
            "date",
            "bool",
            "float",
        } == set(cleaned.columns), "Unexpected columns"
        assert len(dp.num_cols_) == 2
        assert set(dp.num_cols_) == {"float", "int"}
        assert len(dp.bool_cols_) == 1
        assert set(dp.bool_cols_) == {"bool"}
        assert len(dp.cat_cols_) == 1
        assert set(dp.cat_cols_) == {"cat"}

    def test_missing_values(self):
        data = pd.DataFrame(
            {
                "a": ["a", "b", np.nan, "c", "b"],
                "b": [1, 2, 3, 4, np.nan],
                "c": ["2020-01-01", np.nan, "2020-01-03", "2020-01-04", "2020-01-05"],
                "d": [1, 2, 3, 4, 5],
            }
        )
        data["c"] = pd.to_datetime(data["c"])

        # Remove rows
        # Despite having NaNs in 3 rows, we expect only one row to be removed:
        # - "a" is categorical, thus encoded and has no NaNs anymore
        # So we go from rows 5 -> 3, cols 4 -> 6
        dp = DataProcessor(missing_values="remove_rows")
        cleaned = dp.fit_transform(data)
        assert cleaned.shape == (3, 7), f"Did not remove NaNs as expected \n{cleaned}"
        assert not cleaned.isna().values.any(), "DataFrame still contains NaNs"

        # Remove cols
        # Similar argumentation as above.
        dp = DataProcessor(missing_values="remove_cols")
        cleaned = dp.fit_transform(data)
        assert cleaned.shape == (5, 5), "Did not remove NaNs as expected"
        assert not cleaned.isna().values.any(), "DataFrame still contains NaNs"

        # Replace with 0
        dp = DataProcessor(missing_values="zero")
        cleaned = dp.fit_transform(data)
        assert (cleaned.loc[2, ["a_a", "a_b", "a_c"]] == 0).all()
        assert cleaned["b"][4] == 0
        assert not cleaned.isna().values.any(), "DataFrame still contains NaNs"

        # Interpolate
        dp = DataProcessor(missing_values="interpolate")
        cleaned = dp.fit_transform(data)
        assert not cleaned.isna().values.any(), "DataFrame still contains NaNs"

        # Fill with mean
        dp = DataProcessor(missing_values="mean")
        cleaned = dp.fit_transform(data)
        assert (cleaned.loc[2, ["a_a", "a_b", "a_c"]] == 0).all()
        assert cleaned["b"][4] == 2.5
        assert not cleaned.isna().values.any(), "DataFrame still contains NaNs"

    def test_classification_target(self):
        data = pd.DataFrame(
            {
                "a": [2, 2, 1, 1, 2],
                "b": ["class1", "class2", "class1", "class2", "class1"],
            }
        )

        # Numerical not starting at 0
        dp = DataProcessor(target="a")
        cleaned = dp.fit_transform(data)
        assert set(cleaned["a"]) == {0, 1}

        # Categorical
        dp = DataProcessor(target="b")
        cleaned = dp.fit_transform(data)
        assert set(cleaned["b"]) == {0, 1}

    def test_outliers(self):
        x = pd.DataFrame(
            {
                "a": [*(23 * [1]), 1e15],
                "b": np.linspace(0, 1, 24),
                "target": np.linspace(0, 1, 24).tolist(),
            },
            index=pd.MultiIndex.from_product([[0, 1], range(12)]),
        )

        # Clip
        dp = DataProcessor(outlier_removal="clip", target="target")
        xt = dp.fit_transform(x)
        assert xt.values.max() < 1e15, "Outlier not removed"
        assert not xt.isna().values.any(), "NaN found"

        # z-score
        dp = DataProcessor(outlier_removal="z-score", target="target")
        xt = dp.fit_transform(x)
        assert xt.values.max() < 1e15, "Outlier not removed"
        assert not xt.isna().values.any(), "NaN found"
        assert np.isclose(
            dp.transform(pd.DataFrame({"a": [1e14], "b": [1]})).values.max(), 1e14
        )
        assert dp.transform(pd.DataFrame({"a": [1e16], "b": [1]})).values.max() == 1

        # Quantiles
        dp = DataProcessor(outlier_removal="quantiles", target="target")
        xt = dp.fit_transform(x)
        assert xt.max().max() < 1e15, "Outlier not removed"
        assert not xt.isna().any().any(), "NaN found"
        assert dp.transform(pd.DataFrame({"a": [2], "b": [-2]})).values.max() == 0

    def test_constants(self):
        x = pd.DataFrame({"a": [1, 1, 1, 1, 1], "b": [1, 2, 3, 5, 6]})
        dp = DataProcessor(drop_constants=True)
        xt = dp.fit_transform(x)
        assert "a" not in xt.keys(), "Didn't remove constant column"

    def test_dummies(self):
        x = pd.DataFrame({"a": ["a", "b", "c", "b", "c", "a"]})
        dp = DataProcessor()
        xt = dp.fit_transform(x)
        assert "a" not in xt.keys(), "'a' still in keys"
        assert "a_b" in xt.keys(), "a_b missing"
        assert "a_c" in xt.keys(), "a_c missing"
        xt2 = dp.transform(pd.DataFrame({"a": ["a", "c"]}))
        assert np.allclose(
            xt2 - pd.DataFrame({"a_a": [1, 0], "a_b": [0, 0], "a_c": [0, 1]}), 0
        ), "Converted not correct"

    def test_nan_categorical(self):
        # Setup
        dp = DataProcessor()
        data = pd.DataFrame({"a": ["hoi", np.nan, np.nan, np.nan]})
        cleaned = dp.fit_transform(data)

        # Tests
        assert "a" in dp.cat_cols_, "Did not recognize categorical column."
        assert "a_hoi" in cleaned, f"Cat column not properly converted: {list(cleaned)}"

    def test_settings(self):
        target = "target"
        x = pd.DataFrame(
            {
                "a": ["a", "b", "c", "b", "c", "a"],
                "b": [1, 1, 1, 1, 1, 1],
                target: ["a", "b", "c", "b", "c", "a"],
            }
        )
        dp = DataProcessor(target=target, include_output=False, drop_constants=True)
        xt = dp.fit_transform(x)
        assert len(xt.keys()) == x["a"].nunique()

        dp2: DataProcessor = amplo.loads(amplo.dumps(dp))
        xt2 = dp2.transform(x)  # constant column is not dropped, so we do it manually
        assert np.allclose(xt, xt2.drop("b", axis=1))

    def test_pruner(self):
        x = pd.DataFrame({"a": ["a", "b", "c", "b", "c", "a"], "b": [1, 1, 1, 1, 1, 1]})
        dp = DataProcessor()
        dp.fit_transform(x)
        dp.prune_features(["b"])
        assert dp.num_cols_ == ["b"]
        assert dp.cat_cols_ == []

    def test_json_serializable(self):
        x = pd.DataFrame(
            {"a": ["a", "b", "c", "b", "c", "a"], "b": [1, None, 1, 1, 1, 1]}
        )
        for o in ["quantiles", "z-score", "clip", "none"]:
            for mv in ["remove_rows", "remove_cols", "interpolate", "mean", "zero"]:
                dp = DataProcessor(outlier_removal=o, missing_values=mv)
                dp.fit_transform(x)
                amplo.dumps(dp)

    def test_cat_target(self):
        df = pd.DataFrame(
            {"a": ["a", "b", "c", "b", "c", "a"], "b": [1, 2, 3, 4, 5, 6]}
        )
        dp = DataProcessor(target="a")
        xt = dp.fit_transform(df)
        assert "a" in xt
        assert np.allclose(np.array(xt["a"].values), np.array([0, 1, 2, 1, 2, 0]))

    def test_target_encoding(self):
        target = "target"
        y_values = ["a", "b", "c", "b", "c", "a"]
        data = pd.DataFrame({target: y_values})
        dp = DataProcessor(target=target)
        encoded = dp.encode_labels(pl.from_pandas(data), fit=True).to_pandas()
        assert data[target].nunique() == encoded[target].nunique()
        assert encoded[target].min() == 0, "Encoding must start at zero"
        assert pd.api.types.is_integer_dtype(
            encoded[target]
        ), "Encoding must be of dtype `int`"
        decoded = dp.decode_labels(np.array(encoded[target].values))
        assert (
            decoded == y_values
        ).all(), "Decoding does not result in original dataframe"
