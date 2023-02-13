#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting no features at all.
"""

import pandas as pd

from amplo.automl.feature_processing._base import BaseFeatureExtractor

__all__ = ["NopFeatureExtractor"]


class NopFeatureExtractor(BaseFeatureExtractor):
    """
    Feature processor for extracting no features.

    Each input column will be accepted as a feature.
    """

    def fit(self, data: pd.DataFrame):
        # Fitting: accept each feature/column
        self.add_features(data.drop(self.target, axis=1))
        self.is_fitted_ = True

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.target in data:
            return data[self.features_ + [self.target]]
        return data[self.features_]

    def fit_transform(self, data: pd.DataFrame, **fit_params) -> pd.DataFrame:
        return self.fit(data).transform(data)
