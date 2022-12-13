#  Copyright (c) 2022 by Amplo.
from __future__ import annotations

from warnings import warn

import numpy as np
import pandas as pd
from scipy import stats

from amplo.base import BasePredictor, LoggingMixin
from amplo.utils import hist_search


class DataDriftWarning(Warning):
    pass


class DriftDetector(LoggingMixin):
    # todo add second order pdf fit
    # todo add subsequence drift detection

    def __init__(
        self,
        target: str,
        n_bins: int = 500,
        sigma: int = 3,
        with_pdf: bool = False,
        verbose: int = 1,
    ):
        """Detects data drift in streamed input data.

        Supports only  numerical data as off now.
        - Numerical data is check by the discrete distribution
        - Categorical by new columns ()
        - Dates by their recency

        parameters
        ----------
        target : str
        n_bins : int, default =  500
            Number of bins to put the data in
        sigma : int, default = 3
        with_pdf : bool, default = False
        verbose : int, default = 1
        """
        super().__init__(verbose=verbose)
        # Copy kwargs
        self.target = target
        self.n_bins = n_bins
        self.with_pdf = with_pdf
        self.sigma = sigma

        # Initialize
        self.bins_ = {}
        self.output_bins_ = (None, None)
        self.distributions_ = {}
        self.is_fitted_ = False

    def fit(self, data: pd.DataFrame) -> "DriftDetector":
        """
        Fits the class object
        """
        # Numerical
        numeric = data.select_dtypes(include=np.number)  # type: ignore
        self._fit_bins(numeric)
        self._fit_distributions(numeric)

        # Categorical
        # TODO: check for new values

        # Dates
        # TODO: check for recency

        self.is_fitted_ = True
        return self

    def check(self, data: pd.DataFrame | pd.Series):
        """
        Checks a new dataframe for distribution drift.
        """
        violations = []

        # Convert series
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data.to_dict(), index=[0])

        # Numerical
        numeric = data.select_dtypes(include=np.number)  # type: ignore
        violations.extend(self._check_bins(numeric))
        violations.extend(self._check_distributions(numeric))

        return violations

    def fit_output(self, model: BasePredictor, data: pd.DataFrame):
        """
        Additionally to detecting input drift, we should also detect output drift. When
        the distribution of predicted outcomes change, it's often a sign that some under
        laying dynamics are shifting.
        """
        # If it's a classifier and has predict_proba, we use that :)
        if hasattr(model, "predict_proba"):
            prediction = model.predict_proba(data)[:, 1]  # type: ignore
        else:
            prediction = model.predict(data)

        ma, mi = max(prediction), min(prediction)
        y, x = np.histogram(
            prediction,
            bins=self.n_bins,
            range=(mi - (ma - mi) / 10, ma + (ma - mi) / 10),
        )
        self.output_bins_ = (x.tolist(), y.tolist())

    def check_output(self, model: BasePredictor, data: pd.DataFrame, add: bool = False):
        """
        Checks the predictions of a model.
        """
        if not self.is_fitted_:
            raise ValueError("Object not yet fitted.")

        # If it's a classifier and has predict_proba, we use that :)
        if hasattr(model, "predict_proba"):
            prediction = model.predict_proba(data)[:, 1]  # type: ignore
        else:
            prediction = model.predict(data)

        # Unpack predictions
        x, y = self.output_bins_
        assert isinstance(x, list) and isinstance(y, list)

        count_drifts = 0
        for value in prediction:
            ind = hist_search(x, value)
            if ind == -1 or y[ind] <= 0:
                # Drift detected
                count_drifts += 1
        if count_drifts > 0:
            severity = count_drifts / len(prediction) * 100
            warn(f"Output drift detected! Severity: {severity:.2f}%", DataDriftWarning)

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
            "bins": self.bins_,
            "output_bins": self.output_bins_,
            "distributions": self.distributions_,
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
        self.bins_ = weights.get("bins", {})
        self.output_bins_ = weights.get("output_bins", (None, None))
        self.distributions_ = weights.get("distributions", {})
        return self

    def _fit_bins(self, data: pd.DataFrame):
        """
        Fits a histogram on each numerical column.
        """
        # Fit numerical
        for key in data.keys():
            ma, mi = data[key].max(), data[key].min()
            y, x = np.histogram(
                data[key],
                bins=self.n_bins,
                range=(mi - (ma - mi) / 10, ma + (ma - mi) / 10),
            )
            self.bins_[key] = (x.tolist(), y.tolist())

    def _check_bins(self, data: pd.DataFrame, add: bool = False):
        """
        Checks if the current data falls into bins
        """
        violations = []

        for key in data.keys():
            # Get bins
            if key not in self.bins_:
                violations.append(key)
                continue
            x, y = self.bins_[key]

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
                self.bins_[key] = (x, y)

        if len(violations) > 0:
            warn(
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
            for key in data.keys():
                y, x = np.histogram(data[key], normed=True)  # type: ignore
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

                # Select lowest
                self.distributions_[key] = fitted[np.argmin(distances)]

    def _check_distributions(self, data: pd.DataFrame) -> list:
        """
        Checks whether the new data falls within the fitted distributions
        """
        # Init
        violations = []

        if self.with_pdf:
            # Check all numerical columns
            for key in data.keys():
                dist = getattr(stats, self.distributions_[key]["distribution"])
                params = self.distributions_[key]["params"]
                probabilities = dist.pdf(
                    data[key].values, loc=params[-2], scale=params[-1], *params[:-2]
                )

                if any(p < self.sigma for p in probabilities):
                    violations.append(key)
                    continue

            if len(violations) > 0:
                warn(
                    f"Drift detected! {len(violations)} features outside training bins:"
                    f" {violations}",
                    DataDriftWarning,
                )

        return violations

    def add_output_bins(self, old_bins: tuple, prediction: np.ndarray | pd.Series):
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
