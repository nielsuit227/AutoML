from __future__ import annotations

import polars as pl
from tqdm import tqdm

from amplo.automl.feature_processing._base import (
    BaseFeatureExtractor,
    assert_double_index,
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
    target : str | None, optional
        Target column that must be present in data, by default None
    mode : str | None, optional
        Model mode: {"classification", "regression"}, by default "classification"
    window_size : int, optional, default: None
        Determines how many data rows will be collected and summarized by pooling.
        If None, will determine a reasonable window size for the data at hand.
    strategy : {"exhaustive", "random", "smart"}, default: "smart"
        Fitting strategy for feature extraction.
    verbose : int, optional
        Verbisity for logger, by default 0
    """

    ALL_POOL_FUNC_STR = list(POOL_FUNCTIONS)

    def __init__(
        self,
        target: str | None = None,
        mode: str | None = "classification",
        window_size: int | None = None,
        strategy: str = "smart",
        verbose: int = 1,
    ):
        super().__init__(target=target, mode=mode, verbose=verbose)

        # Assert classification or notset
        if self.mode and self.mode != "classification":
            raise NotImplementedError("Only mode 'classification' supported.")

        # Check inputs and set defaults
        check_dtypes(
            ("window_size", window_size, (type(None), int)),
            ("strategy", strategy, str),
        )

        # Set attributes
        self.window_size = window_size
        self.strategy = strategy

        # Subclasses
        self.col_watch: ScoreWatcher | None = None
        self.pool_watch: ScoreWatcher | None = None

    def fit(self, data: pl.DataFrame, index_cols: list[str]):  # type: ignore[override]
        self.fit_transform(data, index_cols)
        return self

    def fit_transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        """Fits pool functions and aggregates"""
        self.logger.info("Fitting feature aggregator.")

        check_data(data, allow_double_underscore=True)
        data, index_cols, _ = assert_double_index(data, index_cols)

        # Select data
        x = data.drop([self.target, *index_cols])
        y = data[self.target]
        index = data.select(index_cols)

        # Initialize
        self.set_window_size(index)
        self.initialize_baseline(x, y)
        assert self.window_size is not None
        assert self._baseline_score is not None

        # Set score watchers
        if self.strategy == "smart":
            self.col_watch = ScoreWatcher(x.columns)
            self.pool_watch = ScoreWatcher(self.ALL_POOL_FUNC_STR)

        # Initialize
        pool_funcs = self.ALL_POOL_FUNC_STR
        data_out = self.pool_target(data, index_cols)
        y_pooled = data_out[self.target]

        for col in tqdm(x.columns):
            if col in (self.target, *index.columns):
                continue

            for func in pool_funcs:
                if self.should_skip_col_func(col, func):
                    continue

                self.logger.debug(f"Fitting: {func}, {col}")

                # Pooling
                feature = pl_pool(data, col, self.window_size, func)[:, -1]
                score = self.calc_feature_score(feature, y=y_pooled)

                # Update score watchers
                if self.strategy == "smart" and self.col_watch and self.pool_watch:
                    self.col_watch.update(col, score, 1)
                    self.pool_watch.update(func, score, 1)

                # Accept feature
                accepted = self.accept_feature(score)
                if accepted:
                    data_out = data_out.with_column(feature)
                    self.add_features(feature.name)

                # Update baseline
                self.logger.debug(
                    f"{func.ljust(25)} {col.ljust(75)} accepted: {accepted}  "
                    f"{score} / {self._baseline_score}"
                )
                self.update_baseline(score)

        self.is_fitted_ = True
        self.logger.info(f"Accepted {data_out.shape[1] - 3} aggregated features.")

        return data_out

    def transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        """Aggregates data"""
        if not self.is_fitted_:
            raise NotFittedError
        assert self.window_size

        data, index_cols, _ = assert_double_index(data, index_cols)

        # Initialize - include pooled target if provided in input data
        if self.target not in data:
            data = data.with_column(pl.lit(0).alias(self.target))
            data_out = self.pool_target(data, index_cols)
            data_out.drop_in_place(self.target)
        else:
            data_out = self.pool_target(data, index_cols)

        # Pooling
        for feature in self.features_:
            col, pool = feature.split("__pool=")
            data_out = data_out.with_column(
                pl_pool(data, col, self.window_size, pool)[:, -1]
            )

        self.logger.info("Transformed features.")
        return data_out

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

    def pool_target(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:
        """
        Pools target data with given window size.

        Parameters
        ----------
        target : pl.DataFrame
            Data to be pooled. Columns 'log', 'index' and target are required.
        index_cols : list[str] | None
            Column names of the double-index. By default ['log', 'index'].

        Returns
        -------
        pl.DataFrame
            Pooled target data.
        """
        data, index_cols, _ = assert_double_index(data, index_cols)

        # Transform and rename back to self.target
        assert self.window_size is not None
        out = pl_pool(data, self.target, self.window_size, "first")
        return out.rename({f"{self.target}__pool=first": self.target})

    def set_window_size(self, index: pl.DataFrame) -> None:
        """
        Sets the window size in case not provided.

        Notes
        -----
        We'll make the window size such that on average there's 5 samples
        Window size CANNOT be small, it significantly slows down the window calculations.

        Parameters
        ----------
        index : pl.DataFrame
            Index of data to be fitted.
        """
        if self.window_size is not None:
            self.logger.debug(f"Window size (from args): {self.window_size}.")
            return

        # Count log sizes
        col_1, col_2 = index.columns
        counts = index.groupby(col_1).count()["count"]
        counts_min: int = counts.min()  # type: ignore[assignment]
        counts_max: int = counts.max()  # type: ignore[assignment]
        counts_mean: float = counts.mean()
        ws = int(min(counts_min, counts_mean // 5))

        # Ensure that window size is an integer and at least 50
        # We're doing fft, less than 50 makes no sense
        self.window_size = max(ws, 50)
        self.logger.debug(f"Set window size to {self.window_size}.")
        if counts_max // self.window_size > 100:
            self.logger.warning("Data with >100 windows will result in slow pooling.")
