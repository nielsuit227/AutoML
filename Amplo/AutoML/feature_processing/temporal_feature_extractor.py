#  Copyright (c) 2022 by Amplo.
"""
Feature processor for extracting temporal features.
"""
import re
from warnings import warn

import numpy as np
import pandas as pd
import pywt
from scipy import signal, stats

from Amplo.AutoML.feature_processing._base import BaseFeatureExtractor, sanitize_dataframe
from Amplo.base.exceptions import ExperimentalWarning
from Amplo.Utils.utils import check_dtypes

__all__ = ["TemporalFeatureExtractor", "pool_single_index", "pool_multi_index"]


def _n_mean_crossings(x):
    """
    Calculates the number of crossings of x on mean.

    A crossing is defined as two sequential values where the first value is lower than
    mean and the next is greater, or vice-versa.

    Parameters
    ----------
    x : np.ndarray or pd.Series
        Time series data to calculate the feature of.

    Returns
    -------
    float
        Number of mean crossings.
    """
    # Reference: https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    positive = np.signbit(x - np.mean(x))
    return np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0].size


def _cid_ce(x, normalize=True):
    """
    Calculates an estimate for a time series complexity.

    Parameters
    ----------
    x : np.ndarray or pd.Series
        Time series data to calculate the feature of.
    normalize : bool
        Whether to z-transform the time series.

    Returns
    -------
    float
        Estimated complexity.
    """
    # Reference: https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html
    if normalize:
        s = np.std(x)
        if s != 0:
            x = (x - np.mean(x)) / s
        else:
            return 0.0

    x = np.diff(x)
    return np.sqrt(np.dot(x, x))


def pool_single_index(data, ws, agg_func):
    """
    Pools series data with given aggregation function.

    Parameters
    ----------
    data : pd.Series
        Data to be pooled.
    ws : int
        Window size; an unsigned integer.
    agg_func : typing.Callable or typing.Dict[str, typing.Callable]
        Function or dictionary of functions to be called for aggregation.
        cf. Examples

    Returns
    -------
    pd.DataFrame
        Pooled data.

    Examples
    --------
    >>> pool_single_index(data, 10, np.mean)
    >>> pool_single_index(data, 10, {"min": np.min, "max": np.max})

    Notes
    -----
    This function assumes the data to be of one time-series.
    """
    # Fill tail when number of missing values in the tail is greater than half
    # the window size.  Otherwise, remove the tail.
    tail = data.shape[0] % ws
    n_missing_in_tail = ws - tail
    if 0 < n_missing_in_tail < ws / 2:
        add_to_tail = data.iloc[-n_missing_in_tail:]
        data = pd.concat([data, add_to_tail])
    elif tail != 0:
        data = data.iloc[:-tail]

    # Assure that agg_func is a dictionary
    if not isinstance(agg_func, dict):
        agg_func = {agg_func.__name__: agg_func}

    # Pooling
    np_data = data.values.reshape(-1, 1, ws)  # noqa
    pooled_data = pd.DataFrame(index=data.index[::ws])
    with np.errstate(divide="ignore", invalid="ignore"):  # ignore true_divide warnings
        for key, func in agg_func.items():
            try:
                pooled_data[key] = func(np_data, axis=-1)
            except TypeError:
                # func doesn't support an axis keyword argument
                # Note: This is a very lot slower but at least gives the correct output.
                pooled_data[key] = np.apply_along_axis(func, axis=-1, arr=np_data)

    return pooled_data


def pool_multi_index(data, ws, agg_func):
    """
    Groupby-pools series data with given aggregation function.

    Parameters
    ----------
    data : pd.Series
        Data to be pooled.
    ws : int
        Window size; an unsigned integer.
    agg_func : typing.Callable or typing.Dict[str, typing.Callable]
        Function or dictionary of functions to be called for aggregation.
        cf. Examples

    Returns
    -------
    pd.Series or pd.DataFrame
        Pooled data.

    Examples
    --------
    # returns pd.Series
    >>> pool_multi_index(data, 10, np.mean)
    # returns pd.DataFrame
    >>> pool_multi_index(data, 10, {"min": np.min, "max": np.max})

    Notes
    -----
    This function makes use of the multi-index.
    """

    pooled_data = data.groupby(level=0).apply(
        lambda group: pool_single_index(group.droplevel(0), ws, agg_func)
    )

    return pooled_data


def _extract_wavelets(series, scales, wavelet, name=None):
    assert isinstance(series, pd.Series)

    # Transform
    # Note that sampling_frequency does not matter.
    coeffs, _ = pywt.cwt(series, scales=scales, wavelet=wavelet)

    # Make dataframe
    columns = [f"{name or series.name}__wav__{wavelet}__{s}" for s in scales]
    x_out = pd.DataFrame(coeffs.real.T, index=series.index, columns=columns)

    return x_out


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor for temporal data.

    Parameters
    ----------
    mode : str
        Model mode: {"classification", "regression"}.
    fit_wavelets : typing.List of str, optional
        Wavelet names for feature exploration (for fitting only).
        Defaults: ["cmor1.5-1.0", "gaus4", "gaus7", "cgau2", "cgau6", "mexh"]
    verbose : int
        Verbosity for logger.

    Notes
    -----
    Valid ``wavelet`` parameters can be found via:
        >>> import pywt
        >>> pywt.wavelist()
    """

    _add_to_settings = ["window_size_", *BaseFeatureExtractor._add_to_settings]

    def __init__(self, mode="notset", fit_wavelets=None, verbose=0):
        warn(
            "TemporalFeatureExtractor is an experimental feature.", ExperimentalWarning
        )
        super().__init__(mode=mode, verbose=verbose)

        if self.mode == "regression":
            # Some notes for implementing regression:
            #  - It does not make sense to pool features and target.
            #  - Wavelet transformations may still add some value.
            msg = (
                "TemporalFeatureExtractor is not ready for regression. "
                "Behavior probably won't meet your expectations!"
            )
            warn(msg, UserWarning)

        if fit_wavelets is None:
            # Set defaults
            fit_wavelets = ["cmor1.5-1.0", "gaus4", "gaus7", "cgau2", "cgau6", "mexh"]
        else:
            check_dtypes([("fit_wavelets", fit_wavelets, list)])
            check_dtypes([("fit_wavelets_item", item, str) for item in fit_wavelets])
        self.fit_wavelets = fit_wavelets

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

        # Calculate baseline scores (w/o taking time into account)
        x_sampled = x.sample(n=min(10_000, y.shape[0]), random_state=93837)
        y_sampled = y.loc[x_sampled.index]
        self._init_feature_baseline_scores(x_sampled, y_sampled)
        del x_sampled, y_sampled

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

        # Apply transformations
        x_out = pd.concat(
            [
                self._transform_raw_features(x),
                self._transform_wav_features(x),
            ],
            axis=1,
        )

        return sanitize_dataframe(x_out[self.features_])

    # ----------------------------------------------------------------------
    # Feature processing

    @property
    def raw_features_(self):
        out = [str(c) for c in self.features_ if not re.search(".+__.+__pool=.+", c)]
        return sorted(out)

    def _fit_transform_raw_features(self, x, y_pooled, update_baseline=True):
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
        self.logger.info("Fitting wavelet-transformed features.")

        # Initialize
        fs = 1.0  # correct sampling frequency only matters for plotting
        wavelets = self.fit_wavelets

        # Extract and score wavelet features for each column
        x_out = []
        all_scores = []
        for col in x:
            # Init extracted features
            col_feats = pd.DataFrame(index=y_pooled.index)

            # Get (local) peak frequencies of power spectral density
            freqs, pxx = signal.welch(x=x[col], fs=fs)
            peak_idx, _ = signal.find_peaks(np.log(pxx), prominence=0.3, distance=10)
            peak_freqs = freqs[peak_idx]

            for wv in wavelets:
                # Use the fact: scale = s2f_const / frequency.
                s2f_const = pywt.scale2frequency(wv, scale=1) * fs
                scales = np.round(s2f_const / peak_freqs, 2)

                # Extract and pool
                feats = (
                    x[col]
                    .groupby(level=0)
                    .apply(_extract_wavelets, scales=scales, wavelet=wv, name=col)
                )
                feats_pooled = self._pool_features(feats, drop_nan_columns=True)

                # Append to col_feats
                col_feats = pd.concat([col_feats, feats_pooled], axis=1)

            # Score and decide which features to accept
            scores = self.select_scores(
                col_feats.apply(self._calc_feature_scores, y=y_pooled, axis=0),
                best_n_per_class=50,
                update_baseline=update_baseline,
            )
            x_out += [col_feats[scores.columns]]
            all_scores += [scores]

        x_out = pd.concat(x_out, axis=1)
        all_scores = pd.concat(all_scores, axis=1)
        if update_baseline:
            self._update_feature_baseline_scores(all_scores)

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
                    .groupby(level=0)
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

    @staticmethod
    def _check_x(x, copy=True, sanitize=True, convert_single_index=False):
        # Call parent checking method
        x_check = BaseFeatureExtractor._check_x(x, copy=copy, sanitize=sanitize)
        # Check multi-index
        n_index_cols = len(x.index.names)
        if n_index_cols == 1 and convert_single_index:
            x_check.index = pd.MultiIndex.from_product([[0], x_check.index])
        elif n_index_cols != 2:
            raise ValueError("Data is not properly multi-indexed.")
        return x_check

    def _set_window_size(self, index):
        # Count log sizes
        counts = pd.Series(index=index).fillna(0).groupby(level=0).count()

        # # We set the window size such that we can expect about 10 - 30 windows
        # # per log.  In the case where each log has the same size, we end up with
        # # 15 windows.
        # shortest, longest = counts.min(), counts.max()
        # window_size = int(shortest / 10 + longest / 30) // 2
        # self.window_size_ = max(1, window_size)

        if self.mode == "classification":
            # We set the window size such that it fits the smallest indices count.
            # However, we want that at least 100 data rows will be present after pool.
            ws = int(counts.min())
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

    def _pool_features(self, data, instruction=None, drop_nan_columns=False):
        """
        Pools data with given window size.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be pooled.
        instruction : typing.Dict[str, list of str], optional
            Instructions for pooling. Each key corresponds the column name and
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
        # Define valid pooling functions.
        # Note: The `**k` in lambda expressions is to support `axis` keyword arguments.
        valid_pools = {
            # --- Basics ---
            "min": np.min,
            "max": np.max,
            "mean": np.mean,
            "std": np.std,
            "kurtosis": stats.kurtosis,
            "skew": stats.skew,
            # --- Characteristics ---
            "entropy": stats.entropy,
            "abs_energy": lambda x: np.dot(x, x),
            "abs_max": lambda x, **k: np.max(np.absolute(x), **k),
            # TODO: Add linear trend feature for `slope` and `stderror`. We have to
            #  adjust our pooling function to support target data as an input.
            #  c.f. tsfresh.feature_extraction.linear_trend
            # "linear_trend_slope": ...,
            # "linear_trend_stderror": ...,
            "n_mean_crossings": _n_mean_crossings,
            # --- Difference ---
            "abs_sum_of_changes": lambda x, **k: np.sum(np.abs(np.diff(x, **k)), **k),
            "mean_of_changes": lambda x, **k: np.mean(np.diff(x, **k), **k),
            "abs_mean_of_changes": lambda x, **k: np.mean(np.abs(np.diff(x, **k)), **k),
            "cid_ce": _cid_ce,
        }

        # Initialize
        if not instruction and self.mode == "regression":
            # Default: Use only mean feature (since window_size=1).
            instruction = {col: ["mean"] for col in data}
        elif not instruction:
            # Default: Apply all instructions on each column.
            instruction = {col: list(valid_pools) for col in data}
        else:
            # Validate
            requested_pools = []
            for col in instruction:
                requested_pools += set(instruction[col])
            invalid_pools = set(requested_pools) - set(valid_pools)
            if invalid_pools:
                msg = f"Invalid pooling names found: {invalid_pools}"
                raise ValueError(msg)

        # Pooling
        pooled_data = []
        for col, instr_lst in instruction.items():
            # Translate pooling instructions
            agg_func = {f"{col}__pool={name}": valid_pools[name] for name in instr_lst}
            # Apply
            pooled_col = pool_multi_index(data[col], self.window_size_, agg_func)
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
            # This lambda function finds most occurring class per window.
            agg_func = lambda x: np.bincount(x, minlength=2).argmax()  # noqa: E731
        elif self.mode == "regression":
            agg_func = np.mean
        else:
            msg = f"Feature processor has an invalid mode: {self.mode}"
            raise AttributeError(msg)

        out = pool_multi_index(target, self.window_size_, agg_func)
        out = out.iloc[:, 0].astype(dtype).rename(target.name)

        return out
