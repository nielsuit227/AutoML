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

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
            self.x, self.y, test_size=0.3, random_state=9276306)

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
        message = ("Performance of a linear model should not exceed the "
                   "performance of the model to observe. "
                   f"Score for linear model: {linear_model_score:.4f}. "
                   f"Score for observed model: {obs_model_score:.4f}.")
        return status_ok, message
