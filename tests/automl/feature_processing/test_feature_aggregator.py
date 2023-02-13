from random import randint

import numpy as np
import pandas as pd

from amplo.automl.feature_processing.feature_aggregator import FeatureAggregator


class TestFeatureAggregator:
    agg: FeatureAggregator

    def setup(self):
        self.agg = FeatureAggregator(target="target")

    def test_fit(self, multiindex_data: pd.DataFrame):
        dft = self.agg.fit_transform(multiindex_data)

        # Fitting tests
        assert self.agg.is_fitted_
        assert len(self.agg.features_) > 0
        assert self.agg.col_watch
        assert self.agg.pool_watch
        assert len(self.agg.col_watch.watch) == len(multiindex_data.keys()) - 1
        assert len(self.agg.pool_watch.watch) == len(self.agg.all_pool_func_str)
        assert all("__pool=" in k for k in self.agg.features_)

        # Transform tests
        assert np.allclose(dft, self.agg.transform(multiindex_data))
        assert isinstance(dft.index, pd.MultiIndex)
        assert "target" in dft
        assert all("__pool=" in k for k in map(str, dft.drop("target", axis=1).keys()))

    def test_set_window_size(self):
        self.agg.reset()
        self.agg.set_window_size(pd.MultiIndex.from_product([[0, 1, 3], range(100)]))
        assert self.agg.window_size == 50
        self.agg.window_size = None  # reset, otherwise taken from args.
        self.agg.set_window_size(pd.MultiIndex.from_product([[0, 1, 3], range(1000)]))
        assert self.agg.window_size == 200

    def test_fit_data_to_window_size(self, multiindex_data):
        self.agg.window_size = 7
        df_t = self.agg.fit_data_to_window_size(multiindex_data)
        assert (df_t.groupby(level=0).count() == 7).all().all()
        self.agg.window_size = 15
        df_t = self.agg.fit_data_to_window_size(multiindex_data)
        assert (df_t.groupby(level=0).count() == 15).all().all()

    def test_pool_target(self, multiindex_data):
        self.agg.window_size = 10
        target_pool = self.agg.pool_target(multiindex_data["target"])
        assert target_pool.sum() / target_pool.count() == multiindex_data[
            "target"
        ].sum() / len(multiindex_data)
        assert np.allclose(
            target_pool.values,
            multiindex_data["target"].groupby(level=0).first().values,
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
        dft = self.agg.fit_transform(multiindex_data)
        assert "a__pool=max" in dft
