import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from Amplo.Utils import histSearch


class DriftDetector:

    def __init__(self,
                 num_cols: list = None,
                 cat_cols: list = None,
                 date_cols: list = None,
                 n_bins: int = 500,
                 sigma: int = 3,
                 with_pdf: bool = False,
                 **kwargs):
        """
        Detects data drift in streamed input data.
        Supports numerical, categorical and datetime variables.
        Due to streamed, we don't check distributions, just bins.
        Categorical simply checks whether it's not a new column
        Datetime simply checks whether the date is recent
        """
        # Copy kwargs
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.date_cols = date_cols
        self.n_bins = n_bins
        self.with_pdf = with_pdf
        self.sigma = sigma

        # Initialize
        self.bins = {}
        self.distributions = {}

    def fit(self, data: pd.DataFrame) -> object:
        """
        Fits the class object
        """
        # Numerical
        self._fit_bins(data)
        self._fit_distributions(data)

        # Categorical

        return self

    def check(self, data: pd.DataFrame):
        """
        Checks a new dataframe for distribution drift.
        """
        violations = []

        # Numerical
        violations.extend(self._check_bins(data))

        return violations

    def get_weights(self) -> dict:
        """
        Gets the weights of the fitted object.
        Useful to save :)
        """
        return {
            'bins': self.bins,
            'distributions': self.distributions
        }

    def load_weights(self, weights: dict):
        """
        Sets the weights of the object to recreate a previously fitted object.

        Parameters
        ----------
        weights:
            bins (dict): Bins dictionary with bins and quantities for all numeric keys
            distributions (dict): Dictionary with fitted distributions for all numeric keys.
        """
        self.bins = weights['bins']
        self.distributions = weights['distributions']

    def _fit_bins(self, data: pd.DataFrame):
        """
        Fits a histogram on each numerical column.
        """
        for key in self.num_cols:
            y, x = np.histogram(data[key], bins=self.n_bins)
            self.bins[key] = (x, y)

    def _check_bins(self, data: pd.DataFrame):
        """
        Checks if the current data falls into bins
        """
        violations = []

        for key in self.num_cols:
            # Get bins
            x, y = self.bins[key]

            # Check bins
            if isinstance(data, pd.DataFrame):
                for v in data[key].values:
                    ind = histSearch(x, v)
                    if ind == -1 or y[ind] <= 0:
                        violations.append(key)
                        break
            elif isinstance(data, pd.Series):
                ind = histSearch(x, data[key])
                if ind == -1 or y[ind] <= 0:
                    violations.append(key)

        if len(violations) > 0:
            print(f"[AutoML] Found {len(violations)} features outside training bins.")

        return violations

    def _fit_distributions(self, data: pd.DataFrame):
        """
        Fits a distribution on each numerical column.
        """
        if self.with_pdf:
            distributions = ["gamma", "beta", "dweibull", "dgamma"]
            distances = []
            fitted = []

            # Iterate through numerical columns
            for key in self.num_cols:
                y, x = np.histogram(data[key], normed=True)
                x = (x + np.roll(x, -1))[:-1] / 2.0     # Get bin means

                # Iterate through distributions
                for distribution in distributions:
                    # Fit & Get PDF
                    dist = getattr(stats, distribution)

                    # Multiple order fit
                    fit = {}
                    params = dist.fit(data[key])
                    fitted_pdf = dist.pdf(x, loc=params[-2], scale=params[-1], *params[:-2])

                    # Analyse
                    distances.append(sum((y - fitted_pdf) ** 2))
                    fitted.append({
                        'distribution': distribution,
                        'params': params,
                    })
                plt.legend(['Original'] + distributions)
                plt.show()

                # Select lowest
                self.distributions[key] = fitted[np.argmin(distances)]

    def _check_distributions(self, data: pd.DataFrame) -> list:
        """
        Checks whether the new data falls within the fitted distributions
        """
        if self.with_pdf:
            # Init
            violations = []

            # Check all numerical columns
            for key in self.num_cols:
                dist = getattr(stats, self.distributions[key]['distribution'])
                params = self.distributions[key]['params']
                probabilities = dist.pdf(data[key].values, loc=params[-2], scale=params[-1], *params[:-2])

                if any(p < self.sigma for p in probabilities):
                    violations.append(key)
                    continue

            if len(violations) > 0:
                print(f"[AutoML] Found {len(violations)} features outside training distribution.")

            return violations
