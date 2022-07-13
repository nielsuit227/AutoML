#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting no features at all.
"""

from amplo.automl.feature_processing._base import BaseFeatureExtractor

__all__ = ["NopFeatureExtractor"]


class NopFeatureExtractor(BaseFeatureExtractor):
    """
    Feature processor for extracting no features.

    Each input column will be accepted as a feature.
    """

    def _fit_transform(self, x, y=None, **fit_params):
        self.add_features(x)
        return self._transform(x=x, y=y)

    def _transform(self, x, y=None):
        return x[self.features_]
