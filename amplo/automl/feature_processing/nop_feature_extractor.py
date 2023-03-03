#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting no features at all.
"""

from __future__ import annotations

import polars as pl

from amplo.automl.feature_processing._base import BaseFeatureExtractor

__all__ = ["NopFeatureExtractor"]


class NopFeatureExtractor(BaseFeatureExtractor):
    """
    Feature processor for extracting no features.

    Each input column will be accepted as a feature.
    """

    def fit(self, data: pl.DataFrame, index_cols: list[str]):  # type: ignore[override]
        # Fitting: accept each feature/column
        self.add_features(data.drop([*index_cols, self.target]))
        self.is_fitted_ = True

        return self

    def transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        if self.target in data:
            return data[[*index_cols, *self.features_, self.target]]
        return data[[*index_cols, *self.features_]]

    def fit_transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        return self.fit(data, index_cols).transform(data, index_cols)
