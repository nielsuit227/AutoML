#  Copyright (c) 2022 by Amplo.

"""
Observer for checking production readiness.
"""
from time import time

from amplo.observation._base import PipelineObserver, _report_obs

__all__ = ["ProductionObserver"]


class ProductionObserver(PipelineObserver):
    """
    Observer before putting to production.

    Testing and monitoring are key considerations for ensuring the production-
    readiness of an ML system, and for reducing technical debt of ML systems.

    Parameters
    ----------
    pipeline : Pipeline
        The amplo pipeline object that will be observed.
    """

    def __init__(self, pipeline):
        super().__init__(pipeline)
        # TODO read below
        # We need this to be executed every time a prediction is made.
        # Should probably use these as wrappers for the predict & predict_proba
        # function to collect the statistics. For now, this class is not used.

    def observe(self):
        # self.check_prediction_latency()
        pass

    @_report_obs
    def check_prediction_latency(self, threshold=0.1):
        """
        Check the latency of predicting a single sample.

        If it takes longer than 100ms, something is wrong.

        Parameters
        ----------
        threshold : float
            Threshold for latency (in s).

        Returns
        -------
        status_ok : bool
            Observation status. Indicates whether a warning should be raised.
        message : str
            A brief description of the observation and its results.
        """
        prediction_latencies = []
        for repeat in range(10):
            t_start = time()
            self.fitted_model.predict(self.x.sample(n=1))
            prediction_latencies.append(time() - t_start)

        status_ok = not any(latency > threshold for latency in prediction_latencies)
        message = (
            f"The latency of predicting a single sample took longer than "
            f"{threshold * 1e3:.0f}ms. Latencies: {prediction_latencies}."
        )

        return status_ok, message
