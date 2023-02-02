import numpy as np
import pandas as pd
import polars as pl

from amplo.automl.feature_processing.pooling import (
    linear_trend,
    linear_trend_bias,
    linear_trend_error,
    n_mean_crossings,
    peak_loc,
    peak_val,
    pl_pool,
)


class TestPooling:
    df = pl.DataFrame(
        {
            "log": [1, 1, 1, 1, 1, 1],
            "index": [0, 1, 2, 3, 4, 5],
            "a": [0, 1.5, 0.5, 3.5, 1.5, 4.5],
        }
    )

    @staticmethod
    def groupby_agg(df: pl.DataFrame, func: pl.Expr, column: str = "a"):
        return (
            df.groupby_dynamic("index", every="50i", by="log")
            .agg(func)
            .select(column)[0, 0]
        )

    def test_n_mean_crossings(self):
        assert (
            self.groupby_agg(self.df, n_mean_crossings("a")) == 4
        )  # +1 from first null (due to diff)

    def test_linear_trend(self):
        assert np.isclose(
            self.groupby_agg(self.df, linear_trend("a")), 0.728, atol=0.001
        )

    def test_linear_trend_bias(self):
        assert np.isclose(
            self.groupby_agg(self.df, linear_trend_bias("a")), -0.633, atol=0.001
        )

    def test_linear_trend_error(self):
        assert np.isclose(
            self.groupby_agg(self.df, linear_trend_error("a")), 5.919, atol=0.001
        )

    def test_peak_detection(self):
        assert self.groupby_agg(self.df, peak_loc("a", 1)) == 1
        assert self.groupby_agg(self.df, peak_loc("a", 2)) == 3
        assert self.groupby_agg(self.df, peak_val("a", 1)) == 1.5
        assert self.groupby_agg(self.df, peak_val("a", 2)) == 3.5

        # Test with the absence of peaks
        df = pl.DataFrame(
            {
                "log": [1, 1, 1, 2, 2, 2],
                "index": [0, 1, 2, 0, 1, 2],
                "a": [0, 0, 0, 0, 0, 0],
            }
        )
        assert self.groupby_agg(df, peak_loc("a", 1)) == -1
        assert self.groupby_agg(df, peak_loc("a", 2)) == -1
        assert self.groupby_agg(df, peak_val("a", 1)) == -1
        assert self.groupby_agg(df, peak_val("a", 2)) == -1

    def test_pl_pool(self):
        pooled = pl_pool(self.df, 3, "abs_max")
        assert pooled.equals(
            pd.Series(
                [1.5, 4.5],
                name="a__pool=abs_max",
                index=pd.MultiIndex.from_product([[1], [0, 3]], names=["log", "index"]),
            )
        )
