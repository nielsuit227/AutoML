#  Copyright (c) 2022 by Amplo.

"""
Observer for checking data.

This part of code is strongly inspired by [1].

References
----------
[1] E. Breck, C. Shanging, E. Nielsen, M. Salib, D. Sculley (2017).
The ML test score: A rubric for ML production readiness and technical debt
reduction. 1123-1132. 10.1109/BigData.2017.8258038.
"""

import json
import re

import numpy as np
import pandas as pd

from Amplo.Observation.base import PipelineObserver, _report_obs
from Amplo.Utils.logging import logger

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
        1. Feature columns should not be monotonically in-/decreasing.
        2. Feature columns should not be sensitive in minority classes
    """

    TYPE = "data_observation"

    def observe(self):
        self.check_monotonic_columns()
        self.check_minority_sensitivity()
        self.check_extreme_values()
        self.check_categorical_mismatch()

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
        logger.info("Checking data for monotonic columns (often indices).")
        x_data = pd.DataFrame(self.x)
        numeric_data = x_data.select_dtypes(include=np.number)

        monotonic_columns = []
        for col in numeric_data.columns:
            series = (
                numeric_data[col].sort_index().interpolate()
            )  # is shuffled when classification
            is_monotonic = series.is_monotonic or series.is_monotonic_decreasing
            is_constant = series.nunique() == 1
            if is_monotonic and not is_constant:
                monotonic_columns.append(col)

        status_ok = not monotonic_columns
        message = (
            f"{len(monotonic_columns)} columns are monotonically in- or "
            f"decreasing. More specifically: {monotonic_columns}"
        )
        return status_ok, message

    @_report_obs
    def check_minority_sensitivity(self):
        """
        Checks whether the data has sensitivities towards minority classes.

        Minority sensitivity is a concept where a signal is present in a small subsample
        of a minority class. As this minority class is potentially not well-covered,
        the small subsample should not be indicative to identify the class or vice versa

        This is analysed by a simple discrete distribution of 10 bins. Minority
        sensitivity is defined as:
        - Bin contains < 2% of total data
        - Bin contains to one class
        - Bin contains > 20% of that class
        (-> class needs to be 10% of data or smaller)

        Note: Only supports numeric columns

        Returns
        -------
        status_ok : bool
            Observation status. Indicates whether a warning should be raised.
        message : str
            A brief description of the observation and its results.
        """
        logger.info("Checking data for minority sensitive columns.")
        minority_sensitive = []

        for key in self.x.keys():
            if not pd.api.types.is_numeric_dtype(self.x[key]):
                # Todo implement for categorical columns
                continue

            # Make bins
            counts, edges = np.histogram(self.x[key].fillna(0), bins=10)

            # Check if a minority group is present
            minority_size = min([c for c in counts if c != 0])
            if minority_size > len(self.x) // 50:
                continue

            # If present, check the labels
            bin_indices = np.digitize(self.x[key], bins=edges)
            for bin_ind in np.where(counts == minority_size)[0]:
                minority_indices = np.where(bin_indices == bin_ind + 1)[0]

                # No minority if spread across labels
                if self.y.iloc[minority_indices].nunique() != 1:
                    continue

                # Not sensitive if only a fraction of the label
                if (
                    len(minority_indices)
                    > (self.y == self.y.iloc[minority_indices[0]]).sum() // 5
                ):
                    minority_sensitive.append(key)

        status_ok = not minority_sensitive
        message = (
            f"{len(minority_sensitive)} columns have minority sensitivity. "
            "Consider to remove them or add data."
            f" More specifically: {minority_sensitive}."
        )
        return status_ok, message

    @_report_obs
    def check_extreme_values(self):
        """
        Checks whether extreme values are present.

        Returns
        -------
        status_ok : bool
            Observation status. Indicates whether a warning should be raised.
        message : str
            A brief description of the observation and its results.
        """
        logger.info("Checking data for extreme values.")
        extreme_values = []

        for key in self.x.keys():
            if not pd.api.types.is_numeric_dtype(self.x[key]):
                # Todo implement for categorical columns
                continue

            if self.x[key].abs().max() > 1000:
                extreme_values.append(key)

        status_ok = not extreme_values
        message = (
            f"{len(extreme_values)} columns have values > 1000. "
            f" More specifically: {extreme_values}."
        )
        return status_ok, message

    @_report_obs
    def check_categorical_mismatch(self):
        """
        Checks whether categorical variables are mismatched

        For example "New York" and "new york". We do this with a simple regex, removing
        all special characters and lowercasing the category.

        Returns
        -------
        status_ok : bool
            Observation status. Indicates whether a warning should be raised.
        message : str
            A brief description of the observation and its results.
        """
        categorical_mismatches = []

        # Loop through all keys
        for key in self.x.keys():
            # Skip if not object
            if not pd.api.types.is_object_dtype(self.x[key]):
                continue

            # Get variants
            variants = self.x[key].unique()

            # Check if one is similar
            for i, variant_x in enumerate(variants):
                for j, variant_y in enumerate(variants):
                    # We only need to compare all once
                    if j >= i:
                        continue

                    # Clean
                    clean_x = re.sub("[^a-z0-9]", "", variant_x.lower())
                    clean_y = re.sub("[^a-z0-9]", "", variant_y.lower())

                    # Compare
                    if clean_x == clean_y:
                        categorical_mismatches.append({key: [variant_x, variant_y]})

        status_ok = not categorical_mismatches
        message = (
            f"{len(categorical_mismatches)} categorical columns have mismatching."
            f"categories. More specifically: {json.dumps(categorical_mismatches)}."
        )
        return status_ok, message
