from itertools import product
from random import randint

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

from amplo.automl.feature_processing.feature_aggregator import FeatureAggregator
from amplo.automl.feature_processing.pooling import pl_pool
from amplo.utils.data import pandas_to_polars


class TestFeatureAggregator:
    agg: FeatureAggregator

    def setup(self):
        self.agg = FeatureAggregator(target="target")

    def test_fit(self, multiindex_data: pd.DataFrame):
        data, index_renaming = pandas_to_polars(multiindex_data)
        index_cols = list(index_renaming)
        dft = self.agg.fit_transform(data, index_cols)

        # Fitting tests
        assert self.agg.is_fitted_
        assert len(self.agg.features_) > 0
        assert self.agg.col_watch
        assert self.agg.pool_watch
        assert len(self.agg.col_watch.watch) == len(multiindex_data.keys()) - 1
        assert len(self.agg.pool_watch.watch) == len(self.agg.ALL_POOL_FUNC_STR)
        assert all("__pool=" in k for k in self.agg.features_)

        # Transform tests
        assert_frame_equal(dft, self.agg.transform(data, index_cols))
        assert set(index_cols).issubset(dft.columns)
        assert "target" in dft
        assert all(
            "__pool=" in k
            for k in map(str, dft.drop([*index_cols, self.agg.target]).columns)
        )

    def test_set_window_size(self):
        self.agg.reset()
        self.agg.set_window_size(pl.DataFrame(product(range(3), range(100))))
        assert self.agg.window_size == 50
        self.agg.window_size = None  # reset, otherwise taken from args.
        self.agg.set_window_size(pl.DataFrame(product(range(3), range(1000))))
        assert self.agg.window_size == 200

    def test_pool_target(self, multiindex_data):
        self.agg.window_size = 10
        data, index_renaming = pandas_to_polars(multiindex_data)
        index_cols = list(index_renaming)
        target_pool = self.agg.pool_target(data, index_cols)[self.agg.target]
        assert (
            target_pool.sum() / target_pool.len()
            == data[self.agg.target].sum() / data.height
        )
        assert_series_equal(
            target_pool,
            pl_pool(data, self.agg.target, self.agg.window_size, "first")[:, -1],
            check_names=False,
        )

    def test_should_pool_max(self, multiindex_data):
        # Make an example where the max pooling would be super beneficial
        # Simply all zeroes, and on a random location a 1.
        # This is not technically the place where to test the pooling functions,
        # They are separately tested. This is more an integration test.
        multiindex_data = multiindex_data[["a", "target"]]
        multiindex_data["a"] = 0
        for i in multiindex_data.index.get_level_values(0).unique():
            multiindex_data.loc[i, randint(0, 9)]["a"] = 1

        data, index_renaming = pandas_to_polars(multiindex_data)
        index_cols = list(index_renaming)
        dft = self.agg.fit_transform(data, index_cols)
        assert "a__pool=max" in dft
