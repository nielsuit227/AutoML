from __future__ import annotations

from warnings import warn

import pandas as pd
import polars as pl
from tqdm import tqdm

from amplo.automl.feature_processing._base import (
    BaseFeatureExtractor,
    assert_multi_index,
    check_data,
)
from amplo.automl.feature_processing.pooling import POOL_FUNCTIONS, pl_pool
from amplo.automl.feature_processing.score_watcher import ScoreWatcher
from amplo.base.exceptions import NotFittedError
from amplo.utils import check_dtypes

__all__ = ["FeatureAggregator"]


class FeatureAggregator(BaseFeatureExtractor):
    """Aggregates a timeseries into a single sample using various pooling functions

    Returns only features deemed worthy, and never the original features.

    NOTE: Only for multi-index classification problems.

    Parameters
    ----------
    window_size : int, optional
        Window size for the aggregation
    verbose : int, default = 1
    """

    all_pool_func_str = list(POOL_FUNCTIONS.keys())
    all_pool_funcs = POOL_FUNCTIONS

    def __init__(
        self,
        target: str = "",
        strategy: str = "smart",
        window_size: int | None = None,
        verbose: int = 1,
    ):
        super().__init__(mode="classification", target=target, verbose=verbose)
        check_dtypes(("strategy", strategy, str))
        self.window_size = window_size
        self.strategy = strategy
        self.col_watch: ScoreWatcher | None = None
        self.pool_watch: ScoreWatcher | None = None

    def fit(self, data: pd.DataFrame):
        self.fit_transform(data)
        return self

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fits pool functions and aggregates"""
        self.logger.info("Fitting feature aggregator.")
        check_data(data, allow_double_underscore=True)
        data, _ = assert_multi_index(data)
        data.index.names = ["log", "index"]
        self.set_window_size(data.index)
        assert self.window_size is not None

        # Set baseline
        x, y = data.drop(self.target, axis=1), data[self.target]
        self.initialize_baseline(x, y)
        assert self._baseline_score is not None

        # Set score watchers
        if self.strategy == "smart":
            self.col_watch = ScoreWatcher(x.keys().tolist())
            self.pool_watch = ScoreWatcher(self.all_pool_func_str)

        # Initialize
        pool_funcs = [p for p in self.all_pool_funcs]
        features: list[pd.Series] = []

        pl_df = pl.from_pandas(pd.concat([data.index.to_frame(), data], axis=1))
        y_pooled = self.pool_target(y)

        for col in tqdm(x):
            col = str(col)
            for func in pool_funcs:
                if self.should_skip_col_func(col, func) or col == self.target:
                    continue
                feature = pl_pool(
                    pl_df.select(["log", "index", col]), self.window_size, func
                )
                score = self.calc_feature_score(feature, y=y_pooled)

                # Update score watchers
                if self.strategy == "smart" and self.col_watch and self.pool_watch:
                    self.col_watch.update(col, score, 1)
                    self.pool_watch.update(func, score, 1)

                # Accept feature
                accepted = self.accept_feature(score)
                if accepted:
                    features.append(feature)
                    self.add_features(str(feature.name))

                # Update baseline
                self.logger.debug(
                    f"{func.ljust(25)} {col.ljust(75)} accepted: {accepted}  "
                    f"{score} / {self._baseline_score}"
                )
                self.update_baseline(score)

        self.is_fitted_ = True
        self.logger.info(f"Accepted {len(features)} aggregated features.")
        return pd.concat(features + [y_pooled], axis=1)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregates data"""
        if not self.is_fitted_:
            raise NotFittedError
        assert self.window_size
        check_data(data, allow_double_underscore=True)
        data, _ = assert_multi_index(data)

        output = []
        pldf = pl.from_pandas(pd.concat([data.index.to_frame(), data], axis=1))

        for feature in self.features_:
            key, pool = feature.split("__pool=")
            output.append(
                pl_pool(pldf.select(["log", "index", key]), self.window_size, pool)
            )

        if self.target in data:
            output.append(self.pool_target(data[self.target]))

        self.logger.info("Transformed features.")
        return pd.concat(output, axis=1)

    def should_skip_col_func(self, col: str, func: str) -> bool:
        """Checks whether current iteration of column / function should be skipped.

        parameters
        ----------
        col : str
        func : str
        """
        # Check score watchers
        if self.strategy == "smart":
            if self.col_watch is None or self.pool_watch is None:
                raise ValueError("Watchers are not set.")
            if self.col_watch.should_skip(col) or self.pool_watch.should_skip(func):
                self.logger.debug(f"Scorewatcher skipped: {func}, {col}")
                return True
        return False

    def pool_target(self, target: pd.Series):
        """
        Pools target data with given window size.

        Parameters
        ----------
        target : pd.Series
            Target data to be pooled.

        Returns
        -------
        pd.Series
            Pooled target data.
        """
        target, _ = assert_multi_index(target, allow_single_index=True)
        target.index.names = ["log", "index"]
        return target.groupby(
            by=[
                target.index.get_level_values("log"),
                target.index.get_level_values("index") // self.window_size,
            ]
        ).first()

    def set_window_size(self, index: pd.Index) -> None:
        """
        Sets the window size in case not provided.

        Notes
        -----
        We'll make the window size such that on average there's 5 samples
        Window size CANNOT be small, it significantly slows down the window calculations.

        Parameters
        ----------
        index : pandas.Index
            Index of data to be fitted.
        """
        if self.window_size is not None:
            self.logger.debug("Window size taken from args.")
            return

        # Count log sizes
        counts = pd.Series(index=index, dtype=int).fillna(0).groupby(level=0).count()
        ws = int(min(counts.min(), counts.mean() // 5))

        # Ensure that window size is an integer and at least 50
        # We're doing fft, less than 50 makes no sense
        self.window_size = max(int(ws), 50)
        self.logger.debug(f"Set window size to {self.window_size}.")
        if counts.max() // self.window_size > 100:
            warn("Data with over a 100 windows, will result in slow pooling.")

    def fit_data_to_window_size(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the data to a multiple of the window size.

        Make sure to always call this function before pooling the data.
        Also, notice that this function uses ``_add_or_remove_tail`` internally.

        Parameters
        ----------
        data : tuple of PandasType
            Data to be fitted to window size.

        Returns
        -------
        data : PandasType or List of PandasType
        """
        self.logger.debug("Fitting data to window size")
        # Check datum
        if len(data.index.names) != 2:
            raise ValueError("Index is not a MultiIndex of size 2.")

        # Add or remove tail
        return data.groupby(level=0, group_keys=False, sort=False).apply(
            self.add_or_remove_tail
        )

    def add_or_remove_tail(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill or cut the tail to fit the data length to a multiple of the window size.

        This is a helper function to be used with ``_fit_data_to_window_size`` and
        treats the data as being single indexed.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fill, cut or leave its tail.

        Returns
        -------
        pd.DataFrame
            Parsed data.
        """
        if self.window_size is None:
            raise ValueError("Window size not yet set.")
        if self.window_size == 1:
            return data

        tail = data.shape[0] % self.window_size
        n_missing_in_tail = self.window_size - tail
        if 0 < n_missing_in_tail < self.window_size / 2:
            # Fill up tail
            add_to_tail = data.iloc[-n_missing_in_tail:]
            data = pd.concat([data, add_to_tail])
        elif tail != 0:
            # Cut tail
            data = data.iloc[:-tail]

        return data
