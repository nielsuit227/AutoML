# Copyright by Amplo
"""
Observer for checking production readiness of model.

This part of code is strongly inspired by [1].

References
----------
[1] E. Breck, C. Shanging, E. Nielsen, M. Salib, D. Sculley (2017).
The ML test score: A rubric for ML production readiness and technical debt
reduction. 1123-1132. 10.1109/BigData.2017.8258038.
"""
import copy

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

from Amplo.Classifiers import PartialBoostingClassifier
from Amplo.Observation.base import PipelineObserver, _report_obs
from Amplo.Regressors import PartialBoostingRegressor

__all__ = ["ModelObserver"]


class ModelObserver(PipelineObserver):
    """
    Model observer before putting to production.

    While the field of software engineering has developed a full range of best
    practices for developing reliable software systems, similar best-practices
    for ML model development are still emerging.

    The following tests are included:
        1. TODO: Model specs are reviewed and submitted.
        2. TODO: Offline and online metrics correlate.
        3. TODO: All hyperparameters have been tuned.
        4. TODO: The impact of model staleness is known.
        5. A simpler model is not better.
        6. TODO: Model quality is sufficient on important data slices.
        7. TODO: The model is tested for considerations of inclusion.
    """

    TYPE = "model_observer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xt, self.xv, self.yt, self.yv = train_test_split(
            self.x, self.y, test_size=0.3, random_state=9276306
        )
        self.fitted_model = self.model
        self.fitted_model.fit(self.xt, self.yt)

    def observe(self):
        self.check_better_than_linear()
        self.check_noise_invariance()
        self.check_slice_invariance()
        self.check_boosting_overfit()

    @_report_obs
    def check_better_than_linear(self):
        """
        Checks whether the model exceeds a linear model.

        This test incorporates the test ``Model 5`` from [1].

        Citation:
            A simpler model is not better: Regularly testing against a very
            simple baseline model, such as a linear model with very few
            features, is an effective strategy both for confirming the
            functionality of the larger pipeline and for helping to assess the
            cost to benefit tradeoffs of more sophisticated techniques.

        Returns
        -------
        status_ok : bool
            Observation status. Indicates whether a warning should be raised.
        message : str
            A brief description of the observation and its results.
        """
        # Make score for linear model
        if self.mode == self.CLASSIFICATION:
            linear_model = LogisticRegression()
        elif self.mode == self.REGRESSION:
            linear_model = LinearRegression()
        else:
            raise AssertionError("Invalid mode detected.")
        linear_model.fit(self.xt, self.yt)
        linear_model_score = self.scorer(linear_model, self.xv, self.yv)

        # Make score for model to observe
        obs_model_score = self.scorer(self.fitted_model, self.xv, self.yv)

        status_ok = obs_model_score > linear_model_score
        message = (
            "Performance of a linear model should not exceed the "
            "performance of the model to observe. "
            f"Score for linear model: {linear_model_score:.4f}. "
            f"Score for observed model: {obs_model_score:.4f}."
        )
        return status_ok, message

    @_report_obs
    def check_noise_invariance(self):
        """
        This checks whether the model performance is invariant to noise in the data.

        Noise is injected in a slice of the data. The noise follows
        the distribution of the original data.
        Next, the performance metrics are re-evaluated on this noisy slice.

        Returns
        -------
        status_ok : bool
            Observation status. Indicates whether a warning should be raised.
        message : str
            A brief description of the observation and its results.
        """
        # Train model
        model = self.model
        model.fit(self.x, self.y)

        # Inject noise
        signal_noise_ratio = 20
        xn = copy.deepcopy(self.xv)
        for key in self.xv.keys():
            signal_energy = sum(self.xv[key] ** 2)
            noise = np.random.normal(0, 1, len(xn))
            noise_energy = sum(noise ** 2)
            xn[key] = (
                self.xv[key]
                + np.sqrt(signal_energy / noise_energy * signal_noise_ratio) * noise
            )

        # Arrange message
        status_ok = True
        message = (
            "Model performance deteriorates with realistic noise injection."
            "This indicates too little variance in your data. "
            "Please upload more varied data."
        )

        # Compare performance
        baseline = self._pipe.scorer(model, self.xv, self.yv)
        comparison = self._pipe.scorer(model, xn, self.yv)
        if comparison / baseline < 0.9 or comparison / baseline > 1.1:
            status_ok = False

        return status_ok, message

    @_report_obs
    def check_slice_invariance(self):
        """
        Model performance should be invariant to data slicing.

        Using High Density Regions [1], the weakest slice of 10% data is identified.
        If the optimization metric is significantly (>5%) worse than the average
        metric, a warning is given.

        [1] https://stats.stackexchange.com/questions/148439/what-is-a-highest-density-region-hdr # noqa

        Returns
        -------
        status_ok : bool
            Observation status. Indicates whether a warning should be raised.
        message : str
            A brief description of the observation and its results.
        """
        # Arrange message
        status_ok = True
        message = (
            "Model performs significantly worse on bad slice of the data. "
            "This indicates too little variance in your data. "
            "Please upload more varied data."
        )

        # Normalize
        x = copy.deepcopy(self.x)
        x -= x.mean()
        x /= x.std()

        # Fit Kernel Density Estimation & get probabilities
        log_probabilities = (
            KernelDensity(kernel="gaussian", bandwidth=1).fit(x).score_samples(x)
        )
        probabilities = np.exp(log_probabilities)

        # Select smallest slice (10%) (selects per class to avoid imbalance)
        if self.mode == "classification":
            slice_indices = []
            for yc in self.y.unique():
                yc_ind = np.where(self.y == yc)[0]
                samples = int(np.ceil(len(yc_ind) // 10))  # Ceils (to avoid 0)
                slice_indices.extend(
                    yc_ind[np.argpartition(probabilities[yc_ind], samples)[:samples]]
                )
        else:
            slice_indices = np.argpartition(probabilities, int(np.ceil(len(x) // 10)))[
                : int(np.ceil(len(x) // 10))
            ]
        train_indices = [i for i in range(len(x)) if i not in slice_indices]
        xt, xv = self.x.iloc[train_indices], self.x.iloc[slice_indices]
        yt, yv = self.y.iloc[train_indices], self.y.iloc[slice_indices]

        # Train and check performance
        model = self.model
        model.fit(xt, yt)
        score = self._pipe.scorer(model, xv, yv)
        if score < self._pipe.best_score:
            status_ok = False

        return status_ok, message

    @_report_obs
    def check_boosting_overfit(self):
        """
        Checks whether boosting models are overfitted.

        Boosting models are often optimal. Though naturally robust against
        overfitting, it's not impossible to add too many estimators in a
        boosting model, creating complexity to an extent of overfitting.
        This function runs the same model while limiting the estimators, and
        checks if the validation performance decreases.

        Returns
        -------
        status_ok : bool
            Observation status. Indicates whether a warning should be raised.
        message : str
            A brief description of the observation and its results.
        """
        # Check if a boosting model has been selected
        if (
            not type(self.model).__name__
            in PartialBoostingClassifier._SUPPORTED_MODELS
            + PartialBoostingRegressor._SUPPORTED_MODELS
        ):
            return True, ""
        if self.mode == "classification":
            PartialBooster = PartialBoostingClassifier
        else:
            PartialBooster = PartialBoostingRegressor

        # Determine steps & initiate results
        steps = np.ceil(
            np.linspace(0, PartialBooster.n_estimators(self.fitted_model), 7)
        )[1:-1]
        scores = []
        for step in steps:
            # Can directly use scorer, as no training is involved at all
            scores.append(
                self.scorer(PartialBooster(self.fitted_model, step), self.xv, self.yv)
            )

        # Now, the check fails if there has been a decrease in performance
        status_ok = all(np.diff(scores) / np.max(np.abs(scores)) > 0.001)
        message = (
            "Boosting overfit detected. Please retrain with less estimators."
            f"Estimators: {steps}, Scores: {scores}"
        )
        return status_ok, message
