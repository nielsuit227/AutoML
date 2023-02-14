#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting temporal features.
"""

from __future__ import annotations

from typing import TypeVar

import pandas as pd

from amplo.automl.feature_processing._base import (
    PERFECT_SCORE,
    BaseFeatureExtractor,
    assert_multi_index,
    check_data,
)
from amplo.automl.feature_processing.feature_aggregator import FeatureAggregator
from amplo.automl.feature_processing.wavelet_extractor import WaveletExtractor
from amplo.base.exceptions import NotFittedError
from amplo.utils.util import check_dtypes, unique_ordered_list

__all__ = ["TemporalFeatureExtractor"]

PandasType = TypeVar("PandasType", pd.Series, pd.DataFrame)


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor for temporal data.

    This is simply a combination of the aggregation of the wavelet extractor with the
    aggregation of the input (raw) features.

    Parameters
    ----------
    target : str, default ""
    mode : {"notset", "classification", "regression"}, optional, default: "notset"
        Model mode.
    window_size : int, optional, default: None
        Determines how many data rows will be collected and summarized by pooling.
        If None, will determine a reasonable window size for the data at hand.
    fit_raw : bool, default: True
        Whether to include pooling from raw features to extract features.
    fit_wavelets : bool or list[str], default = True
        Whether to search for pooled wavelet features.
        If False, wavelets aren't used.
        If true, defaults to ["cmor1.5-1.0", "gaus4", "gaus7", "cgau2", "cgau6", "mexh"]
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
    verbose : int, default: 0
        Verbosity for logger.

    Notes
    -----
    Valid ``wavelet`` parameters can be found via:
    >>> import pywt
    >>> pywt.wavelist()
    """

    def __init__(
        self,
        target: str = "",
        mode: str = "notset",
        window_size: int | None = None,
        fit_raw: bool = True,
        fit_wavelets: list[str] | bool | None = None,
        strategy: str = "smart",
        timeout: int = 1800,
        verbose: int = 0,
    ):
        super().__init__(target=target, mode=mode, verbose=verbose)

        # Warnings
        if self.mode == "regression":
            raise NotImplementedError

        # Check inputs and set defaults
        check_dtypes(
            ("window_size", window_size, (type(None), int)),
            ("fit_raw", fit_raw, bool),
            ("fit_wavelets", fit_wavelets, (bool, list, type(None))),
            ("strategy", strategy, str),
            ("timeout", timeout, int),
        )
        wavelets: list[str]
        if fit_wavelets is None or fit_wavelets is True:
            wavelets = ["cmor1.5-1.0", "gaus4", "gaus7", "cgau2", "cgau6", "mexh"]
        elif not fit_wavelets:
            wavelets = []
        elif isinstance(fit_wavelets, list):  # if not True, must be an iterable
            for item in fit_wavelets:
                check_dtypes(("fit_wavelets__item", item, str))
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
        self.window_size_ = window_size
        self.fit_raw = fit_raw
        self.fit_wavelets = wavelets
        self.strategy = strategy
        self.timeout = timeout
        self.is_fitted_ = False

        # Subclasses
        self.wavelet_extractor = WaveletExtractor(target=self.target, verbose=verbose)
        self.wavelet_aggregator = FeatureAggregator(target=self.target, verbose=verbose)
        self.raw_aggregator = FeatureAggregator(target=self.target, verbose=verbose)

    def fit(self, data: pd.DataFrame, **fit_params):
        # We implement fit_transform because we anyhow transform the data. Therefore,
        # when using fit_transform we don't have to do redundant transformations.
        self.fit_transform(data)
        return self

    def fit_transform(self, data: pd.DataFrame, **fit_params) -> pd.DataFrame:
        """Fits and transforms."""
        # Input checks
        self.logger.info("Fitting temporal feature extractor.")
        check_data(data)
        data, _ = assert_multi_index(data)

        # Initialize fitting
        x, y = data.drop(self.target, axis=1), data[self.target]

        # Calculate baseline scores (w/o taking time into account)
        self.initialize_baseline(x, y)
        assert self._baseline_score is not None
        if self._baseline_score > PERFECT_SCORE:
            self.logger.info("Features are good, we're skipping feature aggregation.")
            self.is_fitted_ = True
            self.skipped_ = True
            return data

        # Fit features
        wav_data = self.wavelet_extractor.fit_transform(data)
        wav_agg_data = self.wavelet_aggregator.fit_transform(wav_data)
        raw_data = self.raw_aggregator.fit_transform(data)

        # Set features
        self.set_features(
            self.wavelet_aggregator.features_ + self.raw_aggregator.features_
        )
        self.is_fitted_ = True

        # Merge two paths together
        return self.concat(wav_agg_data, raw_data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """transforms."""
        # Input checks
        self.logger.info("Transforming data.")
        check_data(data)
        data, got_multi_index = assert_multi_index(data)
        if not self.is_fitted_:
            raise NotFittedError
        if self.skipped_:
            return data

        # Apply transformations
        x_out = self.concat(
            self.wavelet_aggregator.transform(self.wavelet_extractor.transform(data)),
            self.raw_aggregator.transform(data),
        )

        # Return
        if got_multi_index:
            return x_out
        return x_out.set_index(x_out.index.droplevel(0))

    def concat(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Concats the data from the raw and wavelet aggregators.

        This is useful, as both or neither will include the target
        """
        df = pd.concat([df1, df2], axis=1)
        if self.target in df:
            return df[self.features_ + [self.target]]
        return df[self.features_]

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
