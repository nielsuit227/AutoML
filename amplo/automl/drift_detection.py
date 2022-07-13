#  Copyright (c) 2022 by Amplo.

import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from amplo.utils import hist_search


class DataDriftWarning(Warning):
    pass


class DriftDetector:
    # todo add second order pdf fit
    # todo add subsequence drift detection

    def __init__(
        self,
        num_cols: list = None,
        cat_cols: list = None,
        date_cols: list = None,
        n_bins: int = 500,
        sigma: int = 3,
        with_pdf: bool = False,
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
        for key in self.cat_cols:
            self.bins[key] = {}
        for key in self.num_cols:
            self.bins[key] = (None, None)
        self.output_bins = (None, None)
        self.distributions = {}

    def fit(self, data: pd.DataFrame) -> "DriftDetector":
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

        violations.extend(self._check_bins(data))
        violations.extend(self._check_distributions(data))

        return violations

    def fit_output(self, model, data: pd.DataFrame):
        """
        Additionally to detecting input drift, we should also detect output drift. When
        the distribution of predicted outcomes change, it's often a sign that some under
        laying dynamics are shifting.
        """
        assert hasattr(model, "predict"), "Model does not have 'predict' attribute."

        # If it's a classifier and has predict_proba, we use that :)
        if hasattr(model, "predict_proba"):
            prediction = model.predict_proba(data)[:, 1]
        else:
            prediction = model.predict(data)

        ma, mi = max(prediction), min(prediction)
        y, x = np.histogram(
            prediction,
            bins=self.n_bins,
            range=(mi - (ma - mi) / 10, ma + (ma - mi) / 10),
        )
        self.output_bins = (x.tolist(), y.tolist())

    def check_output(self, model, data: pd.DataFrame, add: bool = False):
        """
        Checks the predictions of a model.
        """
        assert hasattr(model, "predict"), "Model does not have 'predict' attribute."

        # If it's a classifier and has predict_proba, we use that :)
        if hasattr(model, "predict_proba"):
            prediction = model.predict_proba(data)[:, 1]
        else:
            prediction = model.predict(data)

        # Check all predictions
        x, y = self.output_bins
        count_drifts = 0
        for value in prediction:
            ind = hist_search(x, value)
            if ind == -1 or y[ind] <= 0:
                # Drift detected
                count_drifts += 1
        if count_drifts > 0:
            severity = count_drifts / len(prediction) * 100
            warnings.warn(
                DataDriftWarning(f"Output drift detected! Severity: {severity:.2f}%")
            )

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
            "bins": self.bins,
            "output_bins": self.output_bins,
            "distributions": self.distributions,
        }

    def load_weights(self, weights):
        """
        Sets the weights of the object to recreate a previously fitted object.

        Parameters
        ----------
        weights : typing.Dict[str, dict or tuple]
            Weights of a (fitted) object.
            Expected keys are:
            - "bins" (dict):
                Bins dictionary with bins and quantities for all numeric keys.
            - "output_bins" (tuple):
                Output bins.
            - "distributions" (dict):
                Fitted distributions for all numeric keys.
        """
        self.bins = weights.get("bins", {})
        self.output_bins = weights.get("output_bins", (None, None))
        self.distributions = weights.get("distributions", {})
        return self

    def _fit_bins(self, data: pd.DataFrame):
        """
        Fits a histogram on each numerical column.
        """
        # Fit numerical
        for key in self.num_cols:
            ma, mi = data[key].max(), data[key].min()
            y, x = np.histogram(
                data[key],
                bins=self.n_bins,
                range=(mi - (ma - mi) / 10, ma + (ma - mi) / 10),
            )
            self.bins[key] = (x.tolist(), y.tolist())

        # Fit categorical
        for key in self.cat_cols:
            self.bins[key] = data[key].value_counts().to_dict()

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
                for value in data[key].values:
                    ind = hist_search(x, value)
                    if ind == -1 or y[ind] <= 0:
                        violations.append(key)
                        break
            elif isinstance(data, pd.Series):
                ind = hist_search(x, data[key])
                if ind == -1 or (
                    y[ind] <= 0
                    and y[max(0, ind - 1)] <= 0
                    and y[min(self.n_bins, ind + 1)] <= 0
                ):
                    violations.append(key)

            # Add data
            if add:
                y += np.histogram(data[key], bins=x)
                self.bins[key] = (x, y)

        if len(violations) > 0:
            warnings.warn(
                DataDriftWarning(
                    f"Drift detected! "
                    f"{len(violations)} features outside training bins: {violations}"
                )
            )

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
                x = (x + np.roll(x, -1))[:-1] / 2.0  # Get bin means

                # Iterate through distributions
                for distribution in distributions:
                    # Fit & Get PDF
                    dist = getattr(stats, distribution)

                    # Multiple order fit
                    params = dist.fit(data[key])
                    fitted_pdf = dist.pdf(
                        x, loc=params[-2], scale=params[-1], *params[:-2]
                    )

                    # Analyse
                    distances.append(sum((y - fitted_pdf) ** 2))
                    fitted.append(
                        {
                            "distribution": distribution,
                            "params": params,
                        }
                    )
                plt.legend(["Original"] + distributions)
                plt.show()

                # Select lowest
                self.distributions[key] = fitted[np.argmin(distances)]

    def _check_distributions(self, data: pd.DataFrame) -> list:
        """
        Checks whether the new data falls within the fitted distributions
        """
        # Init
        violations = []

        if self.with_pdf:
            # Check all numerical columns
            for key in self.num_cols:
                dist = getattr(stats, self.distributions[key]["distribution"])
                params = self.distributions[key]["params"]
                probabilities = dist.pdf(
                    data[key].values, loc=params[-2], scale=params[-1], *params[:-2]
                )

                if any(p < self.sigma for p in probabilities):
                    violations.append(key)
                    continue

            if len(violations) > 0:
                warnings.warn(
                    f"Drift detected! {len(violations)} features outside training bins:"
                    f" {violations}",
                    DataDriftWarning,
                )

        return violations

    def add_output_bins(
        self, old_bins: tuple, prediction: Union[np.ndarray, pd.Series]
    ):
        """
        Just a utility, adds new data to an old distribution.
        """
        if len(old_bins) != 0:
            x, y = old_bins
            yn = np.histogram(prediction, bins=x)[0].tolist()
            y = [y[i] + yn[i] for i in range(len(y))]
        else:
            y, x = np.histogram(prediction, bins=self.n_bins)
            x, y = x.tolist(), y.tolist()
        return x, y

    def add_bins(self, bins: dict, data: pd.DataFrame):
        """
        Just a utility, adds new data to an old distribution.
        """
        # Add numerical bins
        for key in self.num_cols:
            if key in data:
                if key in bins:
                    x, y = bins[key]
                elif key in self.bins:
                    x, _ = self.bins[key]
                    y = [0 for i in range(len(x) - 1)]
                else:
                    raise ValueError(f"Drift Detector - Adding unfitted feature. {key}")
                yn = np.histogram(data[key], bins=x)[0].tolist()
                y = [y[i] + yn[i] for i in range(len(y))]
                bins[key] = (x, y)

        # Add categorical
        for key in self.cat_cols:
            if key in data:
                counts = data[key].value_counts().to_dict()
                bins[key] = {
                    k: counts.get(k, 0) + bins.get(key, {}).get(k, 0)
                    for k in counts.keys() | bins.get(key, {}).keys()
                }
        return bins
