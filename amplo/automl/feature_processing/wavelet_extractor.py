from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pywt
from scipy import signal

from amplo.automl.feature_processing._base import PERFECT_SCORE, BaseFeatureExtractor
from amplo.automl.feature_processing.score_watcher import ScoreWatcher
from amplo.base.exceptions import NotFittedError

__all__ = ["WaveletExtractor"]


class WaveletExtractor(BaseFeatureExtractor):
    """This class extracts wavelets, which carry frequency information.

    Contrary to FFT, wavelets provide a trade-off between temporal and frequency info.
    This class only works for multi-indexed classification data. This would also be
    applicable to regression problems, as the data dimension is unchanged.

    parameters
    ----------
    wavelets : list[str], optional
        Which wavelet families should be tried
    target : str
    strategy : str, default='smart'
        Whether to use the ScoreWatcher, do a random search or exhaustive
    timeout : int
        In seconds
    verbose : int
    """

    def __init__(
        self,
        wavelets: list[str] | None = None,
        target: str = "",
        strategy: str = "smart",
        timeout: int = 1800,
        verbose: int = 1,
    ) -> None:
        super().__init__(target=target, mode="classification", verbose=verbose)
        if strategy not in ("smart", "random", "exhaustive"):
            raise ValueError("Strategy should be 'smart' or 'random'.")

        self.wavelets = (
            wavelets
            if wavelets
            else ["cmor1.5-1.0", "gaus4", "gaus7", "cgau2", "cgau6", "mexh"]
        )
        self.target = target
        self.strategy = strategy
        self.timeout = timeout

        self.peak_freqs_: dict[str, npt.NDArray[Any]] = {}
        self.start_time: float | None = None
        self.col_watch: ScoreWatcher | None = None
        self.wav_watch: ScoreWatcher | None = None

    def fit(self, data: pd.DataFrame):
        """
        It's a trade-off between speed and memory to decide whether we want to directly
        transform or fit first.
        When fitting first, the features can be directly overwritten.
        When transforming directly, we don't have to run the wavelet transform twice.
        The wavelet transform is rather expensive.
        """
        self.fit_transform(data)
        self.is_fitted_ = True
        return self

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Fitting wavelet extractor.")
        x, y = data.drop(self.target, axis=1), data[self.target]

        # Get peak freqs
        self.set_peak_freqs(x)

        # Set baseline
        self.initialize_baseline(x, y)
        assert self._baseline_score is not None
        if self._baseline_score > PERFECT_SCORE:
            self.logger.info("Features are good, we're skipping feature aggregation.")
            self.is_fitted_ = True
            self.skipped_ = True
            return data

        # Initialize data and score tracker
        if self.strategy == "smart":
            self.col_watch = ScoreWatcher(x.keys().to_list())
            self.wav_watch = ScoreWatcher(self.wavelets)
        features = []

        self.start_time = time.time()
        for col, wav in self.get_wavelet_combinations(x):
            if self.should_skip_col_wav(col, wav):
                continue
            if time.time() - self.start_time > self.timeout:
                self.logger.info("Timeout reached, skipping rest.")
                break
            self.logger.debug(f"Fitting: {wav}, {col}")

            # Use the fact: scale = s2f_const / frequency
            wav_features = self.extract_wavelets(x, wav, col)
            wav_scores = wav_features.apply(self.calc_feature_score, y=y, axis=0)

            # Add score
            if self.strategy == "smart" and self.col_watch and self.wav_watch:
                self.col_watch.update(col, wav_scores.sum(), len(wav_scores))
                self.wav_watch.update(wav, wav_scores.sum(), len(wav_scores))

            # Check if good enough and add
            add_scores = self.select_scores(wav_scores)
            features.append(wav_features[add_scores.index])
            self.add_features(add_scores.index.to_list())
            self.logger.debug(
                f"Accepting {len(add_scores)} / {len(wav_scores)} wavelet features for "
                f"{col.ljust(100)}(baseline: {self._baseline_score} / score: "
                f"{max(wav_scores)})"
            )

        self.is_fitted_ = True
        self.logger.info(f"Accepted {len(features)} wavelet-transformed features.")
        return pd.concat(features + [y], axis=1)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise NotFittedError
        if self.skipped_:
            return data

        # Get columns and wavelet info
        x_out = []
        for f in self.features_:
            col, _, wav, scale = f.split("__")
            x_out.append(self.extract_wavelet(data, wav, col, scale=float(scale)))

        if self.target in data:
            x_out.append(data[self.target])
        return pd.concat(x_out, axis=1)

    def extract_wavelets(
        self,
        data: pd.DataFrame,
        wavelet: str,
        column: str,
        scales: list[int] | None = None,
    ) -> pd.DataFrame:
        """Calculates a wavelet.

        parameters
        ----------
        data : pd.DataFrame
        wav : str
        col : str
        """
        if scales is None:
            fs = 1.0
            s2f_const = pywt.scale2frequency(wavelet, scale=1) * fs
            scale_list = np.round(s2f_const / self.peak_freqs_[column], 2)
        else:
            scale_list = scales

        # Transform
        coeffs, _ = pywt.cwt(data[column], scales=scale_list, wavelet=wavelet)

        # Make dataframe
        return pd.DataFrame(
            coeffs.real.T,
            index=data.index,
            columns=[f"{column}__wav__{wavelet}__{s}" for s in scale_list],
        )

    def extract_wavelet(
        self, data: pd.DataFrame, wavelet: str, column: str, scale: int | float
    ) -> pd.Series:
        """Extracts a single wavelet"""
        coeffs, _ = pywt.cwt(data[column], scales=scale, wavelet=wavelet)
        return pd.Series(
            coeffs.real.reshape((-1)),
            index=data.index,
            name=f"{column}__wav__{wavelet}__{scale}",
        )

    def should_skip_col_wav(self, col: str, wav: str) -> bool:
        """Checks whether current iteration of column / function should be skipped.

        parameters
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

    def get_wavelet_combinations(self, data: pd.DataFrame) -> list[tuple[str, str]]:
        """Returns all column - wavelet combinations.

        parameters
        ----------
        data : pd.DataFrame
        """
        rng = np.random.default_rng(236868)
        col_wav_iterator: list[tuple[str, str]] = [
            (col, wav)
            for col in data
            for wav in self.wavelets
            if self.peak_freqs_[col].size > 0
        ]
        if self.strategy in ("random", "smart"):
            rng.shuffle(col_wav_iterator)
        return col_wav_iterator

    def set_peak_freqs(self, data: pd.DataFrame, fs: float = 1.0) -> None:
        """Calculates the frequencies where the PSD has the highest magnitude.

        parameters
        ----------
        data : pd.DataFrame
        fs : float
            Sampling Frequency
        """
        self.peak_freqs_ = {}
        for col in data:
            freqs, pxx = signal.welch(x=data[col], fs=fs)
            if max(pxx) < 1e-3:
                self.peak_freqs_[col] = np.array([])
                continue
            peak_idx, _ = signal.find_peaks(np.log(pxx), prominence=0.3, distance=10)
            self.peak_freqs_[col] = freqs[peak_idx]
