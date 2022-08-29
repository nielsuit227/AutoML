#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting temporal features.
"""

from __future__ import annotations

import re
import time
from typing import Any, List, TypeVar
from warnings import warn

import numpy as np
import numpy.typing
import pandas as pd
import polars as pl
import pywt
from scipy import signal

from amplo.automl.feature_processing._base import (
    BaseFeatureExtractor,
    sanitize_dataframe,
)
from amplo.automl.feature_processing.pooling import get_pool_functions
from amplo.utils import check_dtypes

__all__ = [
    "pl_pool",
    "ScoreWatcher",
    "TemporalFeatureExtractor",
]

PandasType = TypeVar("PandasType", pd.Series, pd.DataFrame)


# ----------------------------------------------------------------------
# Pooling


def pl_pool(
    series: pd.Series,
    window_size: int,
    aggregation: dict[str, pl.Expr | Any],
    *,
    use_multi_index: bool = True,
) -> pd.DataFrame:
    """
    Pools series data with given aggregation functions.

    Parameters
    ----------
    series : pd.Series
        Data to be pooled.
    window_size : int
        Window size for pooling.
    aggregation : dict of {str: pl.Expr or typing.Any}
        Aggregation functions for pooling.
    use_multi_index : bool
        Whether to make use of multi-index if present.
        If False, pooling won't happen in a "groupby" fashion.

    Returns
    -------
    pd.DataFrame
        Pooled data where each column name consists of the original series data name and
        its pooling function name (keys of `aggregation` parameter).
        When for example the series data name is "series" and one `aggregation` key is
        named "min", the resulting column is named "series__pool=min".
    """
    # Integrity checks
    if series.name is None:
        series.name = "series"
    series.index.names = [
        name if isinstance(name, str) else f"index_{i}"
        for i, name in enumerate(series.index.names)
    ]

    # Make polars aggregator
    pl_agg = []
    for name, func in aggregation.items():
        # Check validity of `aggregation`
        if not hasattr(func, "__call__"):
            raise ValueError(f"Aggregator is not callable: {name}")

        # Convert function to polars
        if func.__module__ == "polars.internals.lazy_functions":
            func = func(str(series.name))
        elif func.__module__ == "polars.internals.expr":
            func = func(pl.col(str(series.name)))
        else:
            func = pl.col(str(series.name)).apply(func)

        # Rename output variable and add to `pl_agg`
        func = func.alias(f"{series.name}__pool={name}")
        pl_agg.append(func)

    # Convert series to polars DataFrame
    pl_df = pl.DataFrame(series.reset_index(drop=False))

    # Set up pooling parameters
    is_double_index = len(series.index.names) == 2
    if not is_double_index or not use_multi_index:
        by, index_column = None, pl_df.columns[0]
    else:
        by, index_column = pl_df.columns[:2]

    # Make sure that MultiIndices of 3 or more don't get lost in `agg`
    if len(series.index.names) > 2:
        pl_agg += [pl.col(col).first() for col in series.index.names[2:]]

    # Pool and return
    return (
        pl_df.groupby_dynamic(index_column, every=f"{window_size}i", by=by)
        .agg(pl_agg)
        .to_pandas()
        .set_index(series.index.names)
    )


# ----------------------------------------------------------------------
# Wavelet extraction


def _extract_wavelets(series, scales, wavelet, name=None):
    check_dtypes("series", series, pd.Series)

    # Transform
    # Note that sampling_frequency does not matter.
    coeffs, _ = pywt.cwt(series, scales=scales, wavelet=wavelet)

    # Make dataframe
    columns = [f"{name or series.name}__wav__{wavelet}__{s}" for s in scales]
    x_out = pd.DataFrame(coeffs.real.T, index=series.index, columns=columns)

    return x_out


# ----------------------------------------------------------------------
# Main


class ScoreWatcher:
    """
    Watcher for scores.

    Parameters
    ----------
    keys : list of str
        Keys for the watcher.

    Attributes
    ----------
    watch : dict of {str: tuple of (int, np.ndarray)}
        Keeps track of the counter and score of each watcher key.
    """

    def __init__(self, keys: list[str]):
        check_dtypes("keys", keys, list)
        check_dtypes(("key__item", item, str) for item in keys)
        self.watch: dict[str, (int, int, np.ndarray)] = {
            key: (0, 0, np.array(0)) for key in keys
        }

    def __getitem__(self, key: str) -> (int, np.ndarray):
        """
        Get the counter and score for the given key.

        Parameters
        ----------
        key : str
            Key of the watcher.

        Returns
        -------
        typing.Tuple[int, np.ndarray]
        """
        count, weight, score = self.watch.get(key)
        return count, score

    def __repr__(self):
        """
        Readable string representation of the class.
        """
        return f"{self.__class__.__name__}({sorted(self.watch)})"

    def update(self, key: str, score: np.typing.ArrayLike, weight: int = 1) -> None:
        """
        Update a key of the watcher.

        Parameters
        ----------
        key : str
            Watcher key.
        score : array_like
            Scoring value(s).
        weight : int
            Weight of the score.

        Returns
        -------
        ScoreWatcher
            Updated instance of the watcher.
        """
        check_dtypes(("key", key, str), ("weight", weight, int))
        count, prev_weight, prev_score = self.watch[key]
        new_weight = prev_weight + weight
        new_score = prev_weight * prev_score + weight * np.asarray(score, float)
        new_score /= new_weight
        self.watch[key] = (count + 1, new_weight, np.asarray(new_score, float))

    def mean(self) -> np.ndarray:
        """
        Calculate the mean of all scores.

        Returns
        -------
        np.ndarray
            Mean of all scores.
        """
        scores = [s for counter, _, s in self.watch.values() if counter > 0]
        return np.asarray(scores, float).mean(0)

    def std(self) -> np.ndarray:
        """
        Calculate the standard deviation of all scores.

        Returns
        -------
        np.ndarray
            Standard deviation of all scores.
        """
        scores = [s for counter, _, s in self.watch.values() if counter > 0]
        return np.asarray(scores, float).std(0)


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor for temporal data.

    Parameters
    ----------
    mode : {"notset", "classification", "regression"}, optional, default: "notset"
        Model mode.
    fit_raw : bool, default: True
        Whether to include pooling from raw features to extract features.
    fit_wavelets : bool or list of str, default: True
        Whether to search for pooled wavelet features.
        If True, default wavelets are used for fitting, i.e.
        ["cmor1.5-1.0", "gaus4", "gaus7", "cgau2", "cgau6", "mexh"].
        Each string must be a valid wavelet name (see notes).
    pooling : list of str, optional, default: None
        Pooling functions to be used for feature extraction.
        If None, defaults will be used.
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

    _add_to_settings = ["window_size_", *BaseFeatureExtractor._add_to_settings]

    def __init__(
        self,
        mode="notset",
        fit_raw=True,
        fit_wavelets=True,
        pooling=None,
        strategy="smart",
        timeout=1800,
        verbose=0,
    ):
        super().__init__(mode=mode, verbose=verbose)

        # Warnings
        if self.mode == "regression":
            # Some notes for implementing regression:
            #  - It does not make sense to pool features and target.
            #  - Wavelet transformations may still add some value.
            msg = (
                "TemporalFeatureExtractor is not ready for regression. "
                "Behavior probably won't meet your expectations!"
            )
            warn(msg, UserWarning)

        # Check inputs and set defaults
        check_dtypes(
            ("fit_raw", fit_raw, bool),
            ("fit_wavelets", fit_wavelets, (bool, list)),
            ("pooling", pooling, (type(None), list)),
            ("strategy", strategy, str),
            ("timeout", timeout, int),
        )
        if fit_wavelets is True:
            fit_wavelets = ["cmor1.5-1.0", "gaus4", "gaus7", "cgau2", "cgau6", "mexh"]
        elif fit_wavelets:  # if not True, must be an iterable
            check_dtypes(("fit_wavelets__item", item, str) for item in fit_wavelets)
        pooling = list(get_pool_functions(pooling))  # validate

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
        self.fit_raw = fit_raw
        self.fit_wavelets = fit_wavelets
        self.pooling = pooling
        self.strategy = strategy
        self.timeout = timeout

        # Flags
        self._received_multi_index = True

    def _fit_transform(self, x, y=None, **fit_params):
        self.logger.info("Start fitting data.")

        # Input checks
        x, y = self._check_x_y(x, y)
        numeric_cols = [
            col for col, typ in zip(x, x.dtypes) if np.issubdtype(typ, np.number)
        ]
        if set(numeric_cols) != set(x):
            warn(
                "Handling non-numeric data is (currently) not supported. "
                "Corresponding columns will be ignored.",
                UserWarning,
            )
            x = x[numeric_cols]

        # Initialize fitting
        self._set_validation_model()
        self._set_window_size(x.index)
        x, y = self._fit_data_to_window_size(x, y)

        # Calculate baseline scores (w/o taking time into account)
        rng = np.random.default_rng(384432)
        random_idxs = rng.choice(
            list(range(y.shape[0])), min(10_000, y.shape[0]), replace=False
        )
        self._init_feature_baseline_scores(x.iloc[random_idxs], y.iloc[random_idxs])
        del rng, random_idxs

        # Fit features
        y_pooled = self._pool_target(y)
        x_out = pd.concat(
            [
                self._fit_transform_raw_features(x, y_pooled, update_baseline=True),
                self._fit_transform_wav_features(x, y_pooled, update_baseline=True),
            ],
            axis=1,
        )

        self.logger.info("Finished fitting.")
        return sanitize_dataframe(x_out[self.features_])

    def _transform(self, x, y=None):
        self.logger.info("Transforming data.")

        # Handle input
        x = self._check_x(x, convert_single_index=True)
        x = self._fit_data_to_window_size(x)

        # Apply transformations
        x_out = pd.concat(
            [
                self._transform_raw_features(x),
                self._transform_wav_features(x),
            ],
            axis=1,
        )

        # Sanitize df
        x_out = sanitize_dataframe(x_out[self.features_])

        # Return
        if self._received_multi_index:
            return x_out
        else:
            return x_out.set_index(x_out.index.droplevel(0))

    # ----------------------------------------------------------------------
    # Feature processing

    @property
    def raw_features_(self):
        out = [str(c) for c in self.features_ if not re.search(".+__.+__pool=.+", c)]
        return sorted(out)

    def _fit_transform_raw_features(self, x, y_pooled, update_baseline=True):
        if not self.fit_raw:
            self.logger.info("Skipped fitting raw features.")
            dummy_y = pd.Series(np.zeros(len(x)), index=x.index, dtype=np.int32)
            return pd.DataFrame(index=self._pool_target(dummy_y).index)

        self.logger.info("Fitting raw features.")

        # Pool all features
        x_pooled = self._pool_features(x, drop_nan_columns=True)

        # Score and decide which features to accept
        scores = self.select_scores(
            x_pooled.apply(self._calc_feature_scores, y=y_pooled, axis=0),
            best_n_per_class=50,
            update_baseline=update_baseline,
        )
        x_out = x_pooled[scores.columns]
        self.logger.info(f"Accepted {x_out.shape[1]} raw features.")

        # Add accepted features
        self.add_features(x_out)

        return x_out

    def _transform_raw_features(self, x):
        if not self.raw_features_:
            self.logger.debug("No raw features added.")
            dummy_y = pd.Series(np.zeros(len(x)), index=x.index, dtype=np.int32)
            return pd.DataFrame(index=self._pool_target(dummy_y).index)

        self.logger.info("Transforming raw features.")

        # Pooling
        pool_info = [tuple(c.split("__pool=")) for c in self.raw_features_]
        pool_info = pd.DataFrame(pool_info).groupby(0).agg(list)[1].to_dict()
        x_pooled = self._pool_features(x, pool_info)

        assert set(self.raw_features_) == set(
            x_pooled
        ), "Expected raw features do not match with actual."

        return x_pooled

    @property
    def wav_features_(self):
        out = [str(c) for c in self.features_ if re.search(".+__wav__.+", c)]
        return sorted(out)

    def _fit_transform_wav_features(self, x, y_pooled, update_baseline=True):
        if not self.fit_wavelets:
            self.logger.info("Skipped fitting wavelet-transformed features.")
            dummy_y = pd.Series(np.zeros(len(x)), index=x.index, dtype=np.int32)
            return pd.DataFrame(index=self._pool_target(dummy_y).index)

        self.logger.info("Fitting wavelet-transformed features.")

        # Pre-initialization
        rng = np.random.default_rng(39478)
        fs = 1.0  # correct sampling frequency only matters for plotting

        # Get (local) peak frequencies of power spectral density
        self.logger.debug("Searching (local) peak frequencies of PSD...")
        peak_freqs = {}
        for col in x:
            freqs, pxx = signal.welch(x=x[col], fs=fs)
            peak_idx, _ = signal.find_peaks(np.log(pxx), prominence=0.3, distance=10)
            peak_freqs[col] = freqs[peak_idx]

        # Initialize column-wavelet iterator
        col_wav_iterator = [
            (col, wav)
            for col in x
            for wav in self.fit_wavelets
            if peak_freqs[col].size > 0
        ]
        if self.strategy in ("random", "smart"):
            # Shuffle iterator
            rng.shuffle(col_wav_iterator)

        # Initialize watchers (for smart timeout)
        col_watcher = ScoreWatcher(x.keys().to_list())
        wav_watcher = ScoreWatcher(self.fit_wavelets)
        pool_watcher = ScoreWatcher(self.pooling)

        # Initialize data and score tracker
        not_skipped_counter = 0
        all_feats_pooled = []
        all_scores = []

        self.logger.debug("Starting search for wavelet features...")
        start_time = time.time()

        for counter, (col, wav) in enumerate(col_wav_iterator):
            # Check timeout criterion
            if (
                self.strategy in ("random", "smart")
                and time.time() - start_time > self.timeout
            ):
                self.logger.info(
                    f"TIMEOUT: Stopped search for wavelet-transformed features. "
                    f"Examined {counter}/{len(col_wav_iterator)} wavelet-column "
                    f"combinations ({100 * counter / len(col_wav_iterator):.2f}%) "
                    f"whereas {counter - not_skipped_counter} unpromising were skipped."
                )
                break

            # Check column and wavelet watchers to decide whether to continue or skip
            col_counter, col_score = col_watcher[col]
            wav_counter, wav_score = wav_watcher[wav]
            col_score: np.ndarray
            wav_score: np.ndarray
            if self.strategy == "smart" and (
                (
                    col_counter > 10
                    and (col_score < col_watcher.mean() - col_watcher.std()).all()
                )
                or (
                    wav_counter > 10
                    and all(wav_score < wav_watcher.mean() - wav_watcher.std())
                )
            ):
                self.logger.debug(f"SKIPPED: wav `{wav}`, col `{col}`")
                continue

            self.logger.debug(f"Fitting: wav `{wav}`, col `{col}`")

            # Update counter
            not_skipped_counter += 1

            # Decide which pooling to apply
            pooling_instructions = [i for i in self.pooling]  # copy!
            if self.strategy == "random":
                rng.shuffle(pooling_instructions)
                select_n = len(pooling_instructions) // 3  # 33%
                pooling_instructions = pooling_instructions[:select_n]
            elif self.strategy == "smart":
                pooling_instructions = []
                for pool in self.pooling:
                    pool_counter, pool_score = pool_watcher[pool]
                    pool_score: np.ndarray
                    if pool_counter > 10 and all(
                        pool_score < pool_watcher.mean() - pool_watcher.std()
                    ):
                        continue
                    pooling_instructions.append(pool)

            # Use the fact: scale = s2f_const / frequency
            s2f_const = pywt.scale2frequency(wav, scale=1) * fs
            scales = np.round(s2f_const / peak_freqs[col], 2)

            # Extract features, pool and score
            feats = (
                x[col]
                .groupby(level=0, sort=False)
                .apply(_extract_wavelets, scales=scales, wavelet=wav, name=col)
            )
            feats_pooled = self._pool_features(feats, pooling_instructions, True)
            scores = feats_pooled.apply(self._calc_feature_scores, y=y_pooled, axis=0)

            # Update watchers
            col_watcher.update(col, scores.sum(1), scores.shape[1])
            wav_watcher.update(wav, scores.sum(1), scores.shape[1])
            for pool in pooling_instructions:
                pool_score = scores.filter(regex=f".*pool={pool}$")
                pool_watcher.update(pool, pool_score.sum(1), pool_score.shape[1])
                # A pooled column might be dropped, e.g. when it contains only NaNs.
                # However, we still want to account for it
                pool_watcher.update(pool, 0.0, feats.shape[1] - pool_score.shape[1])

            # Append good scored features
            good_scores = self.select_scores(
                scores,
                best_n_per_class=20,
                # Update baseline every 20th iteration
                update_baseline=update_baseline and not_skipped_counter % 20 == 0,
            )
            all_scores.append(good_scores)
            all_feats_pooled.append(feats_pooled[good_scores.columns])

            # Debug logging
            if not_skipped_counter % 25 == 0:
                self.logger.debug(
                    f"Finished {not_skipped_counter:5.0f}/{len(col_wav_iterator):5.0f} "
                    f"column-wavelet combinations with a mean time of "
                    f"{(time.time() - start_time) / not_skipped_counter:5.2f} seconds."
                )

        # Finish fitting: concatenate all accepted, pooled features
        x_out = pd.concat(all_feats_pooled, axis=1)
        x_out_scores = pd.concat(all_scores, axis=1)
        if update_baseline:
            self._update_feature_baseline_scores(x_out_scores)

        self.logger.info(f"Accepted {x_out.shape[1]} wavelet-transformed features.")

        # Add accepted features
        self.add_features(x_out)

        return x_out

    def _transform_wav_features(self, x):
        if not self.wav_features_:
            self.logger.debug("No wavelet-transformed features added.")
            dummy_y = pd.Series(np.zeros(len(x)), index=x.index, dtype=np.int32)
            return pd.DataFrame(index=self._pool_target(dummy_y).index)

        self.logger.info("Transforming wavelet-transformed features.")

        # Handle wavelet-transform features
        feat_info = []
        for f in self.wav_features_:
            col_name, intermediate = f.split("__wav__")
            wavelet, scale, _ = intermediate.split("__")
            feat_info.append((wavelet, col_name, scale))

        # Group the features by wavelets (moved to index)
        cols = ["wavelet", "col_name", "scales"]
        feat_by_wt = pd.DataFrame(feat_info, columns=cols).groupby("wavelet").agg(list)

        # Extract wavelets
        x_out = []
        for wv, info in feat_by_wt.iterrows():
            columns = sorted(set(info["col_name"]))
            scales = sorted(map(float, set(info["scales"])))

            # Transform and add to list
            for col in columns:
                x_out += [
                    x[col]
                    .groupby(level=0, sort=False)
                    .apply(_extract_wavelets, scales=scales, wavelet=wv, name=col)
                ]
        x_out = pd.concat(x_out, axis=1)

        # Pooling
        pool_info = [tuple(c.split("__pool=")) for c in self.wav_features_]
        pool_info = pd.DataFrame(pool_info).groupby(0).agg(list)[1].to_dict()
        x_pooled = self._pool_features(x_out, pool_info)

        assert set(self.wav_features_) == set(
            x_pooled
        ), "Expected wavelet-transform features do not match with actual."

        return x_pooled

    # ----------------------------------------------------------------------
    # Utils

    def _check_x(self, x, copy=True, sanitize=True, convert_single_index=False):
        # Call parent checking method
        x_check = BaseFeatureExtractor._check_x(x, copy=copy, sanitize=sanitize)
        # Check multi-index
        n_index_cols = len(x.index.names)
        if n_index_cols == 1 and convert_single_index:
            x_check.index = pd.MultiIndex.from_product([[0], x_check.index])
            self._received_multi_index = False
        elif n_index_cols != 2:
            raise ValueError("Data is not properly multi-indexed.")
        return x_check

    def _set_window_size(self, index):
        # Count log sizes
        counts = pd.Series(index=index, dtype=int).fillna(0).groupby(level=0).count()

        if self.mode == "classification":
            # We set the window size such that it fits the smallest indices count.
            # However, we want that at least 100 data rows will be present after pool.
            ws = max(1, int(counts.min()))
            if len(index) // ws < 100:
                ws = len(index) // 100

            # We don't want window sizes < 3. Pooling wouldn't make sense.
            if ws < 3:
                warn(
                    "The given data proposed using a window size smaller than 3 which "
                    "would be problematic. Setting it to 3 anyhow.",
                    UserWarning,
                )
                ws = 3

            self.window_size_ = ws

        elif self.mode == "regression":
            self.window_size_ = 1

        self.logger.debug(f"Set window size to {self.window_size_}.")

    def _fit_data_to_window_size(self, *data):
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

        if len(data) < 1:
            raise ValueError("Got no data.")

        data = list(data)  # make `data` indexable
        for i, datum in enumerate(data):
            # Check datum
            if len(datum.index.names) != 2:
                raise ValueError("Index is not a MultiIndex of size 2.")
            if pd.unique(datum.index.names).size == 1:
                warn("Index names are not unique. Setting them to ['log', 'index'].")
                datum.index.names = ["log", "index"]
            datum_index_names = list(datum.index.names)

            # Add or remove tail
            datum = datum.groupby(level=0).apply(self._add_or_remove_tail)

            # Somehow the groupby-apply function doesn't always duplicate the first
            # index level. Therefore, using `.droplevel(0)` isn't safe enough.
            index_df = datum.index.to_frame()[pd.unique(datum.index.names)]
            index_df = index_df[datum_index_names]  # preserve order
            datum.index = pd.MultiIndex.from_frame(index_df)
            data[i] = datum

        if len(data) == 1:
            return data[0]
        return data

    def _add_or_remove_tail(self, data):
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
        tail = data.shape[0] % self.window_size_
        n_missing_in_tail = self.window_size_ - tail
        if 0 < n_missing_in_tail < self.window_size_ / 2:
            # Fill up tail
            add_to_tail = data.iloc[-n_missing_in_tail:]
            data = pd.concat([data, add_to_tail])
        elif tail != 0:
            # Cut tail
            data = data.iloc[:-tail]

        return data

    def _pool_features(
        self,
        data,
        instruction: dict[str, list[str]] | list[str] | None = None,
        drop_nan_columns=False,
    ):
        """
        Pools data with given window size.

        In order to avoid potential problems, make sure that the `data` parameter was
        handled with ``_fit_data_to_window_size`` first.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be pooled.
        instruction : dict of {str: list of str} or list of str, optional
            Instructions for pooling. Each key corresponds to the column name and
            its value defines the pooling names for the column.
            Default: Will calculate all implemented pools.
        drop_nan_columns : bool
            If false, all columns--no matter how many NaN values they have--will
            be returned. If true, columns with more than 10% of NaN entries will
            be removed.
            You are strongly encouraged to set this parameter only true when you
            are fitting data.

        Returns
        -------
        pd.DataFrame
            Pooled data.
        """
        if not instruction and self.mode == "regression":
            # Use only mean feature (since window_size=1).
            instruction = {col: "mean" for col in data}
        elif not instruction:
            instruction = {col: None for col in data}  # set default
        elif isinstance(instruction, dict):
            for col, instr in instruction.items():
                instruction[col] = list(get_pool_functions(instr))  # validate
        elif isinstance(instruction, list):
            instr = list(get_pool_functions(instruction))  # validate
            instruction = {col: instr for col in data}
        else:
            raise ValueError(f"Invalid instructions: {instruction}")

        # Pooling
        pooled_data = []
        for col, instr in instruction.items():
            agg_func = get_pool_functions(instr)
            # Apply
            pooled_col = pl_pool(data[col], self.window_size_, agg_func)
            pooled_data += [pooled_col]
        pooled_data = pd.concat(pooled_data, axis=1)

        # Sanitize
        if drop_nan_columns:
            rm_mask = pooled_data.isna().sum() > len(pooled_data) / 10
            pooled_data = pooled_data.loc[:, ~rm_mask]

        return sanitize_dataframe(pooled_data)

    def _pool_target(self, target):
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
        assert isinstance(target, pd.Series), "Data must be pandas.Series."
        assert isinstance(self.window_size_, int), "Invalid window size."

        dtype = target.dtype

        if self.mode == "classification":
            agg_func = get_pool_functions("prominent_class")
        elif self.mode == "regression":
            agg_func = get_pool_functions("mean")
        else:
            msg = f"Feature processor has an invalid mode: {self.mode}"
            raise AttributeError(msg)

        out = pl_pool(target, self.window_size_, agg_func)
        out = out.iloc[:, 0].astype(dtype).rename(target.name)

        return out
