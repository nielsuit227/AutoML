import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from Amplo.Utils import histSearch


class DriftDetector:
    # todo add second order pdf fit
    # todo add subsequence drift detection

    def __init__(self,
                 num_cols: list = None,
                 cat_cols: list = None,
                 date_cols: list = None,
                 n_bins: int = 500,
                 sigma: int = 3,
                 with_pdf: bool = False
                 ):
        """
        Detects data drift in streamed input data.
        Supports numerical, categorical and datetime variables.
        Due to streamed, we don't check distributions, just bins.
        Categorical simply checks whether it's not a new column
        Datetime simply checks whether the date is recent
        """
        # Copy kwargs
        self.num_cols = [] if num_cols is None else num_cols
        self.cat_cols = [] if cat_cols is None else cat_cols
        self.date_cols = [] if date_cols is None else date_cols
        self.n_bins = n_bins
        self.with_pdf = with_pdf
        self.sigma = sigma

        # Initialize
        self.bins = {}
        self.output_bins = (None, None)
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

        # Histogram
        violations.extends(self._check_bins(data))

        # todo add distributions

        return violations

    def fit_output(self, model, data: pd.DataFrame):
        """
        Additionally to detecting input drift, we should also detect output drift. When the distribution of predicted
        outcomes change, it's often a sign that some under laying dynamics are shifting.
        """
        assert hasattr(model, 'predict'), "Model does not have 'predict' attribute."

        # If it's a classifier and has predict_proba, we use that :)
        if hasattr(model, 'predict_proba'):
            prediction = model.predict_proba(data)[:, 1]
        else:
            prediction = model.predict(data)

        ma, mi = max(prediction), min(prediction)
        y, x = np.histogram(prediction, bins=self.n_bins, range=(mi - (ma - mi) / 10, ma + (ma - mi) / 10))
        self.output_bins = (x.tolist(), y.tolist())

    def check_output(self, model, data: pd.DataFrame, add: bool = False):
        """
        Checks the predictions of a model.
        """
        assert hasattr(model, 'predict'), "Model does not have 'predict' attribute."

        # If it's a classifier and has predict_proba, we use that :)
        if hasattr(model, 'predict_proba'):
            prediction = model.predict_proba(data)[:, 1]
        else:
            prediction = model.predict(data)

        # Check all predictions
        x, y = self.output_bins
        no_drift = True
        while no_drift:
            for value in prediction:
                ind = histSearch(x, value)
                if ind == -1 or y[ind] <= 0:
                    logging.warning(f"[AutoML] Output drift detected!")
                    no_drift = False

        # Add new output
        if add:
            y += np.histogram(prediction, bins=x)
            return y

    def get_weights(self) -> dict:
        """
        Gets the weights of the fitted object.
        Useful to save :)
        """
        return {
            'bins': self.bins,
            'output_bins': self.output_bins,
            'distributions': self.distributions,
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
        self.bins = weights['bins'] if 'bins' in weights else dict()
        self.output_bins = weights['output_bins'] if 'output_bins' in weights else (None, None)
        self.distributions = weights['distributions'] if 'distributions' in weights else dict()
        return self

    def _fit_bins(self, data: pd.DataFrame):
        """
        Fits a histogram on each numerical column.
        """
        for key in self.num_cols:
            ma, mi = data[key].max(), data[key].min()
            y, x = np.histogram(data[key], bins=self.n_bins, range=(mi - (ma - mi) / 10, ma + (ma - mi) / 10))
            self.bins[key] = (x.tolist(), y.tolist())

    def _check_bins(self, data: pd.DataFrame, add: bool = False):
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
                if ind == -1 or (y[ind] <= 0 and y[min(0, ind - 1)] <= 0 and y[max(self.n_bins, ind + 1)] <= 0):
                    violations.append(key)

            # Add data
            if add:
                y += np.histogram(data[key], bins=x)
                self.bins[key] = (x, y)

        if len(violations) > 0:
            logging.warning(f"[AutoML] Drift detected!  {len(violations)} features outside training bins: {violations}")

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
                logging.warning(f"[AutoML] Drift detected!  {len(violations)} features outside training bins: "
                                f"{violations}")

            return violations

    @staticmethod
    def add_output_bins(old_bins: tuple, prediction: np.ndarray):
        """
        Just a utility, adds new data to an old distribution.
        """
        x, y = old_bins
        y += np.histogram(prediction, bins=x)
        return x, y

    @staticmethod
    def add_bins(bins: dict, data: pd.DataFrame):
        """
        Just a utility, adds new data to an old distribution.
        """
        for key in data.keys():
            x, y = bins[key]
            y += np.histogram(data[key], bins=x)
            bins[key] = (x, y)
        return bins
