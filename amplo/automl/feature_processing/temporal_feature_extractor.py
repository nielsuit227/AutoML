#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting temporal features.
"""

from __future__ import annotations

import polars as pl

from amplo.automl.feature_processing._base import (
    PERFECT_SCORE,
    BaseFeatureExtractor,
    assert_double_index,
    check_data,
)
from amplo.automl.feature_processing.feature_aggregator import FeatureAggregator
from amplo.automl.feature_processing.wavelet_extractor import WaveletExtractor
from amplo.base.exceptions import NotFittedError
from amplo.utils.util import check_dtypes, unique_ordered_list

__all__ = ["TemporalFeatureExtractor"]


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor for temporal data.

    This is simply a combination of the aggregation of the wavelet extractor with the
    aggregation of the input (raw) features.

    Parameters
    ----------
    target : str | None, optional
        Target column that must be present in data, by default None
    mode : str | None, optional
        Model mode: {"classification", "regression"}, by default None
    window_size : int, optional, default: None
        Determines how many data rows will be collected and summarized by pooling.
        If None, will determine a reasonable window size for the data at hand.
    fit_raw : bool, default: True
        Whether to include pooling from raw features to extract features.
    fit_wavelets : bool | list[str] | None, optional
        Whether to search for pooled wavelet features, by default None.
        If False, wavelets aren't used.
        If True, defaults to ["cmor1.5-1.0", "gaus4", "gaus7", "cgau2", "cgau6", "mexh"]
        If None, same as if True.
        A custom list of wavelets can also be provided
        Each string must be a valid wavelet name (see notes).
    strategy : {"exhaustive", "random", "smart"}, default: "smart"
        Fitting strategy for feature extraction.
        If "exhaustive", use brute-force method.
        If "random", iterates on randomly shuffled feature-wavelet combinations and
        performs pooling on a random subset of `self.pooling` until end of iterator or
        timeout is reached.
        If "smart", similar to "random" but (1) skips unpromising features or wavelets
        and (2) uses promising poolings only.
    timeout : int, default: 1800
        Timeout in seconds for fitting. Has no effect when `strategy` is "exhaustive".
    verbose : int, optional
        Verbisity for logger, by default 0

    Notes
    -----
    Valid ``wavelet`` parameters can be found via:
    >>> import pywt
    >>> pywt.wavelist()
    """

    def __init__(
        self,
        target: str | None = None,
        mode: str | None = None,
        window_size: int | None = None,
        fit_raw: bool = True,
        fit_wavelets: bool | list[str] | None = None,
        strategy: str = "smart",
        timeout: int = 1800,
        verbose: int = 0,
    ):
        super().__init__(target=target, mode=mode, verbose=verbose)

        # Assert classification or notset
        if self.mode and self.mode != "classification":
            raise NotImplementedError("Only mode 'classification' supported.")

        # Check inputs and set defaults
        check_dtypes(
            ("window_size", window_size, (type(None), int)),
            ("fit_raw", fit_raw, bool),
            ("fit_wavelets", fit_wavelets, (type(None), bool, list)),
            ("strategy", strategy, str),
            ("timeout", timeout, int),
        )
        wavelets: list[str] | None
        if fit_wavelets is False:
            wavelets = []  # disable fitting wavelets
        elif fit_wavelets is True:
            wavelets = None
        else:
            wavelets = fit_wavelets

        # Integrity checks
        if strategy not in ("exhaustive", "random", "smart"):
            raise ValueError(f"Invalid value for `strategy`: {strategy}")
        if timeout <= 0:
            raise ValueError(f"`timeout` must be strictly positive but got: {timeout}")
        if not any([fit_raw, fit_wavelets]):
            raise ValueError(
                "Disabling all fitting functions is useless. Enable at least one feature extractor."
            )

        # Set attributes
        self.window_size = window_size
        self.fit_raw = fit_raw
        self.fit_wavelets = wavelets
        self.strategy = strategy
        self.timeout = timeout
        self.is_fitted_ = False

        # Subclasses
        self.wavelet_extractor = WaveletExtractor(
            target=target,
            mode=mode,
            wavelets=wavelets,
            strategy=strategy,
            timeout=timeout,
            verbose=verbose,
        )
        self.wavelet_aggregator = FeatureAggregator(
            target=target,
            mode=mode,
            window_size=window_size,
            strategy=strategy,
            verbose=verbose,
        )
        self.raw_aggregator = FeatureAggregator(
            target=target,
            mode=mode,
            window_size=window_size,
            strategy=strategy,
            verbose=verbose,
        )

    def fit(self, data: pl.DataFrame, index_cols: list[str]):  # type: ignore[override]
        # We implement fit_transform because we anyhow transform the data. Therefore,
        # when using fit_transform we don't have to do redundant transformations.
        self.fit_transform(data, index_cols)
        return self

    def fit_transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        """Fits and transforms."""
        # Input checks
        self.logger.info("Fitting temporal feature extractor.")
        check_data(data)

        # Assert double-index
        assert_double_index(data, index_cols)

        # Split data
        x = data.drop([self.target, *index_cols])
        y = data[self.target]

        # Initialize fitting
        self.initialize_baseline(x, y)
        assert self._baseline_score is not None
        if self._baseline_score > PERFECT_SCORE:
            self.logger.info("Features are good, we're skipping feature aggregation.")
            self.is_fitted_ = True
            self.skipped_ = True
            return data

        # Fit-transform
        raw_agg_data = self.raw_aggregator.fit_transform(data, index_cols)
        wav_data = self.wavelet_extractor.fit_transform(data, index_cols)
        wav_agg_data = self.wavelet_aggregator.fit_transform(wav_data, index_cols)

        data_out = pl.concat(
            [raw_agg_data, wav_agg_data.drop([*index_cols, self.target])],
            how="horizontal",
        )

        self.set_features(
            self.wavelet_aggregator.features_ + self.raw_aggregator.features_
        )

        self.is_fitted_ = True
        return data_out[[*index_cols, *self.features_, self.target]]

    def transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        """Transforms."""

        self.logger.info("Transforming data.")

        if not self.is_fitted_:
            raise NotFittedError
        if self.skipped_:
            return data

        # Input checks
        data, index_cols, got_double_index = assert_double_index(
            data, index_cols, allow_single=True
        )
        check_data(data)

        # Apply transformations
        data_out = pl.concat(
            [
                self.raw_aggregator.transform(data, index_cols),
                self.wavelet_aggregator.transform(
                    self.wavelet_extractor.transform(data, index_cols), index_cols
                ).drop([*index_cols, self.target]),
            ],
            how="horizontal",
        )

        # Restore input indexing
        if not got_double_index:
            data.drop_in_place(index_cols.pop(0))

        return data_out[[*index_cols, *self.features_, self.target]]

    def set_features(self, features: str | list[str]):
        """Updates the features of the aggregators nad extractor.

        Parameters
        ----------
        features : list[str]
        """
        if isinstance(features, str):
            features = [features]

        self.features_ = features
        self.raw_aggregator.set_features([f for f in features if "__wav__" not in f])
        self.wavelet_aggregator.set_features([f for f in features if "__wav__" in f])
        self.wavelet_extractor.set_features(
            unique_ordered_list(
                [f.split("__pool")[0] for f in features if "__wav__" in f]
            )
        )
