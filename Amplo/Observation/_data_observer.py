# Copyright by Amplo
"""
Observer for checking data.

This part of code is strongly inspired by [1].

References
----------
[1] E. Breck, C. Shanging, E. Nielsen, M. Salib, D. Sculley (2017).
The ML test score: A rubric for ML production readiness and technical debt
reduction. 1123-1132. 10.1109/BigData.2017.8258038.
"""

import numpy as np
import pandas as pd

from Amplo.Observation.base import PipelineObserver
from Amplo.Observation.base import _report_obs

__all__ = ["DataObserver"]


class DataObserver(PipelineObserver):
    """
    Data observer before pushing to production.

    Machine learning systems differ from traditional software-based systems in
    that the behavior of ML systems is not specified directly in code but is
    learned from data. Therefore, while traditional software can rely on unit
    tests and integration tests of the code, here we attempt to add a sufficient
    set of tests of the data.

    The following tests are included:
        1. TODO: Feature expectations are captured in a schema.
        2. TODO: All features are beneficial.
        3. TODO: No feature's cost is too much.
        4. TODO: Feature adhere to meta-level requirements.
        5. TODO: The data pipeline has appropriate privacy controls.
        6. TODO: New features can be added quickly.
        7. TODO: All input feature code is tested.
        8. Feature columns should not be monotonically in-/decreasing.
    """

    TYPE = "data_observation"

    def observe(self):
        self.check_monotonic_columns()

    @_report_obs
    def check_monotonic_columns(self):
        """
        Checks whether any column is monotonically in- or decreasing.

        Returns
        -------
        status_ok : bool
            Observation status. Indicates whether a warning should be raised.
        message : str
            A brief description of the observation and its results.
        """
        x_data = pd.DataFrame(self.x)
        numeric_data = x_data.select_dtypes(include=np.number)

        monotonic_columns = []
        for col in numeric_data.columns:
            series = numeric_data[col].sort_index()  # is shuffled when classification
            if series.is_monotonic or series.is_monotonic_decreasing:
                monotonic_columns.append(col)

        status_ok = not monotonic_columns
        message = (f"{len(monotonic_columns)} columns are monotonically in- or "
                   f"decreasing. More specifically: {monotonic_columns}")
        return status_ok, message
