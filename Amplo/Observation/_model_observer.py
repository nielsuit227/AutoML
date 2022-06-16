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

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

from Amplo.Observation.base import PipelineObserver
from Amplo.Observation.base import _report_obs

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

    def observe(self):
        self.check_better_than_linear()

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
        obs_model = self.model
        obs_model.fit(self.xt, self.yt)
        obs_model_score = self.scorer(obs_model, self.xv, self.yv)

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
            avg_val_db = 10 * np.log10(self.xv[key].mean())
            noise_val = 10 ** ((avg_val_db - signal_noise_ratio) / 10)
            xn[key] = self.xv[key] + np.random.normal(0, np.sqrt(noise_val), len(xn))

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

        # Select smallest slice (10%)
        slice_indices = np.argpartition(probabilities, len(x) // 10)[: len(x) // 10]
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
