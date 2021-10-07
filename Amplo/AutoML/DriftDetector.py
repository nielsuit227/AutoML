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

        # Initialize
        self.bins = {}

    def fit(self, data: pd.DataFrame) -> object:
        """
        Fits the class object
        """
        # Numerical
        self._fit_bins(data)

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

    def _fit_bins(self, data: pd.DataFrame):
        """
        Fits bins on the training data.
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
        return violations
