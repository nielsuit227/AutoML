# Copyright by Amplo
"""
Observer for checking production readiness.
"""

from typing import List

from Amplo.Observation.base import BaseObserver
from Amplo.Observation.base import PipelineObserver
from Amplo.Observation._data_observer import DataObserver
from Amplo.Observation._model_observer import ModelObserver

__all__ = ["ProductionObserver"]


class ProductionObserver(BaseObserver):
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
        super().__init__()

        # Set observers
        self._observers: List[PipelineObserver] = [
            DataObserver(pipeline=pipeline),
            ModelObserver(pipeline=pipeline),
        ]

    def observe(self):
        for obs in self._observers:
            # Make observation
            obs.observe()
            # Gather observations
            self.observations.extend(obs.observations)
