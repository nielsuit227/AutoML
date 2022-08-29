#  Copyright (c) 2022 by Amplo.

import time

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    OneSidedSelection,
    RandomUnderSampler,
    TomekLinks,
)
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from amplo.utils.logging import logger


class DataSampler:
    def __init__(
        self,
        method: str = "both",
        margin: float = 0.1,
        cv_splits: int = 3,
        shuffle: bool = True,
        fast_run: bool = False,
        objective: str = "neg_log_loss",
        verbosity: int = 1,
    ):
        """
        Trial and error various strategies:
        - Under-sample all to minority
        - Over-sample all to majority
        - Combination of the two, with say 1k, 5k, 25k, 50k, 75k and 100k samples
        This sensitivity is problem dependent, but has major consequence for hyper
        parameter optimization. Need to get it right.
        Need to make sure that metrics are evaluated on original distribution.
        https://imbalanced-learn.org/stable/references/index.html#api

        Parameters
        ----------
        method : str
            Whether to under-sample, over-sample, or a combination from the both
            ("under", "over", or "both")
        margin : float
            If the imbalance is smaller than this margin, skip the balancing, should be
            within [0, 1]
        fast_run : bool
            If true, skips exhaustive samplers
        objective : str
            Scikit-Learn metric
        """
        # Tests
        if method not in ("under", "over", "both"):
            raise ValueError("Method must be one of 'under', 'over' or 'both'.")
        if not (1e-6 < margin < 1):
            raise ValueError("Margin must be within [1e-6, 1].")

        # Objective
        self.objective = objective or "neg_log_loss"
        self.scorer = metrics.get_scorer(self.objective)

        # Parse
        self.method = method
        self.margin = margin
        self.fast_run = fast_run

        # Settings
        self.cv = StratifiedKFold(n_splits=cv_splits, shuffle=shuffle)
        self.min_samples = 2e2
        self.max_samples = 1e5
        self.verbosity = verbosity

        # Results
        self.is_fitted = False
        self.opt_sampler = None
        self.results = pd.DataFrame(
            columns=[
                "sampler",
                "mean_objective",
                "std_objective",
                "worst_case",
                "mean_time",
                "std_time",
            ]
        )

    def get_samplers(self, y: pd.Series) -> list:
        """
        Return selection of samplers based on the label vector
        Parameters
        ----------
        labels [pd.Series]: Vector with labels
        """
        # Assess labels
        n_classes = y.nunique(dropna=True)
        n_samples = len(y)
        imbalance = y.value_counts().values / n_samples
        n_min_samples = y.value_counts().min() - 2
        logger.info(
            "Data Balancer: Found class balance: {}.".format(
                ", ".join(["{:.1f}%".format(i * 100) for i in imbalance])
            )
        )

        # If imbalance is negligible, return nothing
        if np.min(imbalance) > (1 - self.margin) / n_classes:
            return []

        # Init
        samplers = []

        # Under Samplers
        if self.method != "over" and n_samples > self.min_samples:
            samplers.append(
                OneSidedSelection(
                    sampling_strategy="not minority", n_seeds_S=int(n_samples / 10)
                )
            )
            samplers.append(RandomUnderSampler(sampling_strategy="not minority"))
            if not self.fast_run:
                samplers.append(TomekLinks(sampling_strategy="not minority"))

        # Over Samplers
        if self.method != "under" and n_samples < self.max_samples:
            samplers.append(RandomOverSampler(sampling_strategy="not majority"))
            if not self.fast_run:
                samplers.append(
                    SMOTE(
                        sampling_strategy="not majority",
                        k_neighbors=min(n_min_samples - 1, 6),
                    )
                )
                samplers.append(
                    BorderlineSMOTE(
                        sampling_strategy="not majority",
                        k_neighbors=min(n_min_samples - 1, 5),
                        m_neighbors=min(n_min_samples, 10),
                    )
                )

        if (
            self.method == "both"
            and self.min_samples < n_samples < self.max_samples
            and not self.fast_run
        ):
            samplers.append(
                SMOTETomek(
                    sampling_strategy="all",
                    smote=SMOTE(
                        sampling_strategy="all", k_neighbors=min(n_min_samples, 6)
                    ),
                )
            )
            samplers.append(
                SMOTEENN(
                    sampling_strategy="all",
                    enn=EditedNearestNeighbours(
                        sampling_strategy="all", n_neighbors=min(n_min_samples, 3)
                    ),
                    smote=SMOTE(
                        sampling_strategy="all", k_neighbors=min(n_min_samples, 6)
                    ),
                )
            )

        return samplers

    def _score(self, sampler, x: pd.DataFrame, y: pd.Series) -> [list, list]:
        """
        Scores a sampler for data

        Parameters
        ----------
        sampler : Imbalance-learn sampler
        x [pd.DataFrame]: Input data
        y [pd.Series]: Output data

        Returns
        -------
        Scores [list]: List of objective for every fold
        Times [list]: List of runtimes for every fold
        """
        # Initiate
        scores, times = [], []

        # Cross-Validate
        for i, (ti, vi) in enumerate(self.cv.split(x, y)):
            # Split data
            xt, xv, yt, yv = x.iloc[ti], x.iloc[vi], y.iloc[ti], y.iloc[vi]

            # Note start time
            t_start = time.time()

            # Re - sample
            if sampler is not None:
                x_res, y_res = sampler.fit_resample(xt, yt)
            else:
                x_res, y_res = xt, yt

            # Train a model
            model = RandomForestClassifier(n_estimators=50, max_depth=5)
            model.fit(x_res, y_res)

            # Score
            scores.append(self.scorer(model, xv, yv))
            times.append(time.time() - t_start)

        return scores, times

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """

        Parameters
        ----------
        x [pd.DataFrame]: Input data
        y [pd.Series]: Output data
        """
        # Get samplers
        samplers = self.get_samplers(y)

        # If samplers are empty, we don"t need sampling
        if len(samplers) == 0:
            self.opt_sampler = None

        else:
            if self.verbosity > 0:
                logger.info("Balancing data with {:.0f} samplers".format(len(samplers)))

            # Get baseline
            scores, times = self._score(None, x, y)
            self.results = self.results.append(
                {
                    "sampler": "Baseline",
                    "mean_objective": np.mean(scores),
                    "std_objective": np.std(scores),
                    "worst_case": np.mean(scores) - np.std(scores),
                    "mean_time": np.mean(times),
                    "std_time": np.std(scores),
                },
                ignore_index=True,
            )
            if self.verbosity > 0:
                logger.info(
                    "Fitted {} {}: {:.4f} \u00B1 {:.4f}".format(
                        "Baseline".ljust(25),
                        self.objective,
                        np.mean(scores),
                        np.std(scores),
                    )
                )

            # Iterate through samplers
            for sampler in samplers:
                name = type(sampler).__name__

                # Score
                scores, times = self._score(sampler, x, y)

                # Store results
                self.results = self.results.append(
                    {
                        "sampler": name,
                        "mean_objective": np.mean(scores),
                        "std_objective": np.std(scores),
                        "worst_case": np.mean(scores) - np.std(scores),
                        "mean_time": np.mean(times),
                        "std_time": np.std(times),
                    },
                    ignore_index=True,
                )

                # Print results
                if self.verbosity > 0:
                    logger.info(
                        "Fitted {} {}: {:.4f} \u00B1 {:.4f}".format(
                            name.ljust(25),
                            self.objective,
                            np.mean(scores),
                            np.std(scores),
                        )
                    )

            # Set optimal sampler
            opt_sampler = self.results.loc[
                self.results["worst_case"] == self.results["worst_case"].max(),
                "sampler",
            ].iloc[0]
            if opt_sampler != "Baseline":
                self.opt_sampler = [
                    s for s in samplers if type(s).__name__ == opt_sampler
                ][0]

        # Set fitted
        self.is_fitted = True

    def resample(self, x: pd.DataFrame, y: pd.Series) -> [pd.DataFrame, pd.Series]:
        """
        Resamples the data with the best sampler

        Parameters
        ----------
        x [pd.DataFrame]: Input data
        y [pd.Series]: Output data

        Returns
        -------
        x_res [pd.DataFrame]: Resampled input data
        y_res [pd.DataFrame]: Resampled output data
        """
        # Check whether fitted
        assert self.is_fitted, "Data Sampler not yet fitted. Run .fit(x, y) first. "

        # Check if we need sampling at all
        if self.opt_sampler is None:
            logger.info("No balancing needed :)")
            return x, y

        else:

            # Print start
            if self.verbosity > 0:
                logger.info(
                    "Resampling data with {}".format(type(self.opt_sampler).__name__)
                )

            # Resample
            x_res, y_res = self.opt_sampler.fit_resample(x, y)

            return x_res, y_res

    def fit_resample(self, x: pd.DataFrame, y: pd.Series):
        """
        Fits and returns resampled data

        Parameters
        ----------
        x [pd.DataFrame]: Input data
        y [pd.Series]: Output data

        Returns
        -------
        x_res [pd.DataFrame]: Resampled input data
        y_res [pd.Series]: Resampled output data
        """
        # Fit
        self.fit(x, y)

        # Resample
        x_res, y_res = self.resample(x, y)

        # Return
        return x_res, y_res
