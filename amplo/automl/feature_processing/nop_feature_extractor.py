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
        self.reset()
        x, _ = self._check_data(data, require_y=False)

        # Fitting: accept each feature/column
        self.add_features(x)
        self._is_fitted = True

        return self

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.check_is_fitted()
        x, _ = self._check_data(data, require_y=False)

        # Transforming: select 'fitted' features
        return x[self.features_]

    def transform_target(self, y: pd.Series) -> pd.Series:
        self.check_is_fitted()
        return self._check_y(y, copy=False)
