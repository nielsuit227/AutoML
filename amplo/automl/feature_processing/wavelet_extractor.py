from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pywt
from scipy import signal

from amplo.automl.feature_processing._base import (
    PERFECT_SCORE,
    BaseFeatureExtractor,
    assert_double_index,
)
from amplo.automl.feature_processing.score_watcher import ScoreWatcher
from amplo.base.exceptions import NotFittedError
from amplo.utils.util import check_dtypes

__all__ = ["WaveletExtractor"]


class WaveletExtractor(BaseFeatureExtractor):
    """This class extracts wavelets, which carry frequency information.

    Contrary to FFT, wavelets provide a trade-off between temporal and frequency info.
    This class only works for multi-indexed classification data. This would also be
    applicable to regression problems, as the data dimension is unchanged.

    Parameters
    ----------
    target : str | None, optional
        Target column that must be present in data, by default None
    mode : str | None, optional
        Model mode: {"classification", "regression"}, by default "classification"
    wavelets : list[str] | None, optional
        List of wavelets to choose from.
        If None, defaults to ["cmor1.5-1.0", "gaus4", "gaus7", "cgau2", "cgau6", "mexh"].
        Each string must be a valid wavelet name (see notes), by default None
    strategy : str, optional
        Fitting strategy for feature extraction, by default "smart"
        If "random", iterates on randomly shuffled feature-wavelet combinations and
        performs pooling on a random subset of `self.pooling` until end of iterator or
        timeout is reached.
        If "smart", similar to "random" but (1) skips unpromising features or wavelets
        and (2) uses promising poolings only.
    timeout : int, optional
        Timeout in seconds for fitting, by default 1800
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
        mode: str | None = "classification",
        wavelets: list[str] | None = None,
        strategy: str = "smart",
        timeout: int = 1800,
        verbose: int = 0,
    ) -> None:
        super().__init__(target=target, mode=mode, verbose=verbose)

        # Assert classification or notset
        if self.mode and self.mode != "classification":
            raise NotImplementedError("Only mode 'classification' supported.")

        # Check inputs and set defaults
        check_dtypes(
            ("wavelets", wavelets, (type(None), list)),
            ("strategy", strategy, str),
            ("timeout", timeout, int),
        )
        if wavelets is None:
            wavelets = ["cmor1.5-1.0", "gaus4", "gaus7", "cgau2", "cgau6", "mexh"]
        else:
            check_dtypes(("wavelets__item", item, str) for item in wavelets)
        if strategy not in ("smart", "random", "exhaustive"):
            raise ValueError("Strategy should be 'smart' or 'random'.")
        if timeout <= 0:
            raise ValueError(f"Timeout must be strictly positive but got: {timeout}")

        # Set attributes
        self.wavelets = wavelets
        self.strategy = strategy
        self.timeout = timeout

        self.peak_freqs_: dict[str, npt.NDArray[Any]] = {}
        self.start_time: float | None = None
        self.col_watch: ScoreWatcher | None = None
        self.wav_watch: ScoreWatcher | None = None

    def fit(self, data: pl.DataFrame, index_cols: list[str]):  # type: ignore[override]
        """
        Fit the wavelet extractor on the data.

        Notes
        -----
        It's a trade-off between speed and memory to decide whether we want to directly
        transform or fit first.
        When fitting first, the features can be directly overwritten.
        When transforming directly, we don't have to run the wavelet transform twice.
        The wavelet transform is rather expensive.
        """
        self.fit_transform(data, index_cols)
        return self

    def fit_transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        self.logger.info("Fitting wavelet extractor.")

        data, index_cols, _ = assert_double_index(data, index_cols)

        # Select data
        x = data.drop([self.target, *index_cols])
        y = data[self.target]

        # Set baseline
        self.initialize_baseline(x, y)
        assert self._baseline_score is not None
        if self._baseline_score > PERFECT_SCORE:
            self.logger.info("Features are good, we're skipping feature aggregation.")
            self.is_fitted_ = True
            self.skipped_ = True
            return data

        # Set score watchers
        if self.strategy == "smart":
            self.col_watch = ScoreWatcher(x.columns)
            self.wav_watch = ScoreWatcher(self.wavelets)

        # Initialize
        self.set_peak_freqs(x)
        features: list[pl.DataFrame] = []

        self.start_time = time.time()
        for col, wav in self.get_wavelet_combinations(x):
            if self.should_skip_col_wav(col, wav):
                continue
            if time.time() - self.start_time > self.timeout:
                self.logger.info("Timeout reached, skipping rest.")
                break

            self.logger.debug(f"Fitting: {wav}, {col}")

            # Extract wavelets
            wav_features = self.extract_wavelets(x, wav, col)
            wav_scores: dict[str, float] = {}
            for c in wav_features.columns:
                wav_scores[c] = self.calc_feature_score(wav_features[c], y)

            # Add score
            if self.strategy == "smart" and self.col_watch and self.wav_watch:
                self.col_watch.update(col, sum(wav_scores.values()), len(wav_scores))
                self.wav_watch.update(wav, sum(wav_scores.values()), len(wav_scores))

            # Check if good enough and add
            selected_cols = self.select_scores(wav_scores)
            features.append(wav_features.select(selected_cols))
            self.add_features(selected_cols)
            self.logger.debug(
                f"Accepted {len(selected_cols)} / {len(wav_scores)} wavelet features "
                f"for {col.ljust(100)} (baseline: {self._baseline_score} / score: "
                f"{max(wav_scores)})"
            )

        self.is_fitted_ = True
        self.logger.info(f"Accepted {len(features)} wavelet-transformed features.")
        return pl.concat([data[index_cols], *features, y.to_frame()], how="horizontal")

    def transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        if not self.is_fitted_:
            raise NotFittedError
        if self.skipped_:
            return data

        data, index_cols, _ = assert_double_index(data, index_cols)

        # Get columns and wavelet info
        x_out = []
        for f in self.features_:
            col, _, wav, scale = f.split("__")
            x_out.append(self.extract_wavelet(data, wav, col, scale=float(scale)))

        if self.target in data:
            x_out.append(data[self.target])

        return pl.concat([data[index_cols], pl.DataFrame(x_out)], how="horizontal")

    def extract_wavelets(
        self,
        data: pl.DataFrame,
        wavelet: str,
        column: str,
        scales: list[float] | None = None,
    ) -> pl.DataFrame:
        """Calculates a wavelet.

        Parameters
        ----------
        data : pl.DataFrame
        wav : str
        col : str
        """
        if scales is None:
            # Use the fact: scale = s2f_const / frequency
            fs = 1.0
            s2f_const = pywt.scale2frequency(wavelet, scale=1) * fs
            scales = np.round(s2f_const / self.peak_freqs_[column], 2).tolist()
            assert isinstance(scales, list)

        # Transform and return
        coeffs, _ = pywt.cwt(data[column].to_numpy(), scales=scales, wavelet=wavelet)
        columns = [f"{column}__wav__{wavelet}__{scale}" for scale in scales]
        return pl.from_numpy(coeffs.real, columns=columns, orient="col")

    def extract_wavelet(
        self, data: pl.DataFrame, wavelet: str, column: str, scale: int | float
    ) -> pl.Series:
        """Extracts a single wavelet"""
        df = self.extract_wavelets(data, wavelet, column, [scale])
        return df[df.columns[0]]

    def should_skip_col_wav(self, col: str, wav: str) -> bool:
        """Checks whether current iteration of column / function should be skipped.

        Parameters
        ----------
        col : str
        func : str
        """
        # Check score watchers
        if self.strategy == "smart":
            if self.col_watch is None or self.wav_watch is None:
                raise ValueError("Watchers are not set.")
            if self.col_watch.should_skip(col) or self.wav_watch.should_skip(wav):
                self.logger.debug(f"Scorewatcher skipped: {wav}, {col}")
                return True
        return False

    def get_wavelet_combinations(self, data: pl.DataFrame) -> list[tuple[str, str]]:
        """Returns all column - wavelet combinations.

        parameters
        ----------
        data : pl.DataFrame
        """
        rng = np.random.default_rng(236868)
        col_wav_iterator: list[tuple[str, str]] = [
            (col, wav)
            for col in data.columns
            for wav in self.wavelets
            if self.peak_freqs_[col].size > 0
        ]
        if self.strategy in ("random", "smart"):
            rng.shuffle(col_wav_iterator)
        return col_wav_iterator

    def set_peak_freqs(self, data: pl.DataFrame, fs: float = 1.0) -> None:
        """Calculates the frequencies where the PSD has the highest magnitude.

        parameters
        ----------
        data : pl.DataFrame
        fs : float
            Sampling Frequency
        """
        self.peak_freqs_ = {}
        for col in data.columns:
            freqs, pxx = signal.welch(x=data[col], fs=fs)
            if max(pxx) < 1e-3:
                self.peak_freqs_[col] = np.array([])
                continue
            peak_idx, _ = signal.find_peaks(np.log(pxx), prominence=0.3, distance=10)
            self.peak_freqs_[col] = freqs[peak_idx]
