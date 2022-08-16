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

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.model_selection import cross_val_predict

from amplo.observation._base import PipelineObserver, _report_obs
from amplo.utils.logging import logger
from amplo.utils.metrics import levenshtein_distance

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

    _obs_type = "data_observation"

    def observe(self):
        self.check_monotonic_columns()
        self.check_minority_sensitivity()
        self.check_extreme_values()
        self.check_categorical_mismatch()
        self.check_label_issues()

    @_report_obs
    def check_monotonic_columns(self):
        """
        Checks whether any column is monotonically in- or decreasing.

        When data is multi-indexed it checks monotonicity per group (level 0).

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

        # Make multi-indexed if not already
        if len(numeric_data.index.names) == 1:
            numeric_data.index = pd.MultiIndex.from_product([[0], numeric_data.index])

        # Sort index from the innermost to outermost index level since it is shuffled
        # in classification.
        for axis in list(range(len(numeric_data.index.names)))[::-1]:
            numeric_data.sort_index(axis=axis, inplace=True)

        monotonic_columns = []
        for col in numeric_data.columns:
            grouped_series = (
                numeric_data[col]
                .groupby(level=0)
                .apply(lambda group: group.interpolate(limit_directions="both"))
                .groupby(level=0)
            )
            is_monotonic = (
                grouped_series.apply(lambda group: group.is_monotonic)
                | grouped_series.apply(lambda group: group.is_monotonic_decreasing)
            ).any()
            is_constant = grouped_series.apply(lambda group: group.nunique() == 1).any()
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
        logger.info("Checking data for categorical mismatches.")
        categorical_mismatches = []

        # Get dummy information from pipeline
        for feature, variants in self._pipe.data_processor.dummies.items():

            # Check if one is similar
            for i, variant_x in enumerate(variants):
                for j, variant_y in enumerate(variants):
                    # We only need to compare all once
                    if j >= i:
                        continue

                    # Remove feature from variants
                    variant_x = variant_x[len(feature) + 1 :]
                    variant_y = variant_y[len(feature) + 1 :]

                    # Get distance (normalized)
                    distance = levenshtein_distance(variant_x, variant_y) / max(
                        len(variant_x), len(variant_y)
                    )

                    # Compare
                    if 0 < distance <= 0.2:
                        categorical_mismatches.append({feature: [variant_x, variant_y]})

        status_ok = not categorical_mismatches
        message = (
            f"{len(categorical_mismatches)} categorical columns have mismatching."
            f"categories. More specifically: {json.dumps(categorical_mismatches)}."
        )
        return status_ok, message

    @_report_obs
    def check_label_issues(self):
        """
        Checks for label issues (classification only). Uses cleanlabs.
        """
        # Only classification
        if self._pipe.mode != "classification":
            return True, "All good"

        # Get predicted labels
        predicted_probabilities = cross_val_predict(
            self.model, self.x, self.y, cv=3, method="predict_proba"
        )

        # Analyse with cleanlabs
        ranked_label_issues = find_label_issues(
            labels=self.y,
            pred_probs=predicted_probabilities,
            return_indices_ranked_by="self_confidence",
        ).tolist()

        # Return
        status_ok = not ranked_label_issues
        message = (
            f"{len(ranked_label_issues)} sample(s) seem incorrectly labelled."
            f"More specifically: {json.dumps(ranked_label_issues)}."
        )
        return status_ok, message
