#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting and selecting features.
"""

from __future__ import annotations

import re
from warnings import warn

import numpy as np
import pandas as pd
import psutil

from amplo.automl.feature_processing._base import BaseFeatureExtractor, check_data
from amplo.automl.feature_processing.feature_selection import FeatureSelector
from amplo.automl.feature_processing.nop_feature_extractor import NopFeatureExtractor
from amplo.automl.feature_processing.static_feature_extractor import (
    StaticFeatureExtractor,
)
from amplo.automl.feature_processing.temporal_feature_extractor import (
    TemporalFeatureExtractor,
)
from amplo.base import BaseTransformer, LoggingMixin
from amplo.base.exceptions import NotFittedError
from amplo.utils import check_dtypes

__all__ = [
    "find_collinear_columns",
    "translate_features",
    "get_required_columns",
    "FeatureProcessor",
]


def find_collinear_columns(
    data: pd.DataFrame, information_threshold: float = 0.9
) -> list[str]:
    """
    Finds collinear features and returns them.

    Calculates the Pearson Correlation coefficient for all input features.
    Features that exceed the information threshold are considered linearly
    co-dependent, i.e. describable by: y = a * x + b. As these features add
    little to no information, they will be removed.

    Parameters
    ----------
    data : pd.DataFrame
        Data to search for collinear features.
    information_threshold : float
        Percentage value that defines the threshold for a ``collinear`` feature.

    Returns
    -------
    list of str
        List of collinear feature columns.
    """
    check_dtypes(
        ("data", data, pd.DataFrame),
        ("information_threshold", information_threshold, float),
    )

    # Get collinear features
    nk = data.shape[1]
    corr_mat = np.zeros((nk, nk))

    try:
        # Check available memory and raise error if necessary
        mem_avail = psutil.virtual_memory().available
        mem_data = data.memory_usage(deep=True).sum()
        if mem_avail < 2 * mem_data:
            raise MemoryError(
                "Data is too big to handle time efficient. Using memory efficient "
                "implementation instead."
            )

        # More efficient in terms of time but may crash when data size is huge
        norm_data = (data - data.mean(skipna=True, numeric_only=True)).to_numpy()
        ss = np.sqrt(np.sum(norm_data**2, axis=0))

        for i in range(nk):
            for j in range(nk):
                if i >= j:
                    continue
                sum_ = np.sum(norm_data[:, i] * norm_data[:, j])
                with np.errstate(invalid="ignore"):  # ignore division by zero (out=nan)
                    corr_mat[i, j] = abs(sum_ / (ss[i] * ss[j]))

    except MemoryError:
        # More redundant calculations but more memory efficient
        for i, col_name_i in enumerate(data):
            col_i = data[col_name_i]
            norm_col_i = (col_i - col_i.mean(skipna=True)).to_numpy()
            del col_i
            ss_i = np.sqrt(np.sum(norm_col_i**2))

            for j, col_name_j in enumerate(data):
                if i >= j:
                    continue

                col_j = data[col_name_j]
                norm_col_j = (col_j - col_j.mean(skipna=True)).to_numpy()
                del col_j
                ss_j = np.sqrt(np.sum(norm_col_j**2))

                sum_ = np.sum(norm_col_i * norm_col_j)
                with np.errstate(invalid="ignore"):  # ignore division by zero (out=nan)
                    corr_mat[i, j] = abs(sum_ / (ss_i * ss_j))

    # Set collinear columns
    mask = np.sum(corr_mat > information_threshold, axis=0) > 0
    collinear_columns = np.array(data.columns)[mask].astype(str).tolist()
    return collinear_columns


def translate_features(feature_cols: list[str]) -> dict[str, list[str]]:
    """
    Translates (extracted) features and tells its underlying original feature.

    Parameters
    ----------
    feature_cols : list of str
        Feature columns to be translated.

    Returns
    -------
    dict of {str: list of str}
        Dictionary with `feature_cols` as keys and their underlying original features
        as values.
    """
    for item in feature_cols:
        check_dtypes(("feature_cols__item", item, str))

    translation = {}
    for feature in feature_cols:
        # Raw features
        if "__" not in feature:
            t = [feature]
        # From StaticFeatureExtractor
        elif re.search("__(mul|div|x|d)__", feature):
            f1, _, f2 = feature.split("__")
            t = [f1, f2]
        elif re.search("^(sin|cos|inv)__", feature):
            _, f = feature.split("__")
            t = [f]
        # From TemporalFeatureExtractor
        elif re.search("^((?!__).)*__pool=.+", feature):  # `__` appears only once
            f, _ = feature.split("__")
            t = [f]
        elif re.search(".+__wav__.+__pool=.+", feature):
            f, _ = feature.split("__", maxsplit=1)
            t = [f]
        else:
            raise ValueError(f"Could not translate feature: {feature}")

        translation[feature] = t

    return translation


def get_required_columns(feature_cols: list[str]) -> list[str]:
    """
    Returns all required columns that are required for the given features.

    Parameters
    ----------
    feature_cols : list of str
        Feature columns to be translated.

    Returns
    -------
    list[str]
        All required data columns for the given features.
    """

    required_cols = []
    for translation in translate_features(feature_cols).values():
        required_cols.extend(translation)

    return sorted(set(required_cols))


class FeatureProcessor(BaseTransformer, LoggingMixin):
    """
    Feature processor module to extract and select features.

    Parameters
    ----------
    target : str
        Target column that must be present in data.
    mode : "classification", "regression"
        Model mode.
    is_temporal : bool, optional
        Whether the data should be treated as temporal data or not.
        If none is provided, is_temporal will be set to true when fit data is
        multi-indexed, false otherwise.
    extract_features : bool
        Whether to extract features or just remove correlating columns.
    collinear_threshold : float
        Information threshold for collinear features.
    analyse_feature_sets : {"auto", "all", "gini", "shap"}, default: "auto"
        Which feature sets to analyse.
        If "auto", gini (and shap) will be analysed.
        If "all", gini and shap will be analysed.
        If "gini" or "shap", gini or shap will be analysed, respectively.
    selection_cutoff : float
        Upper feature importance threshold for threshold feature selection.
    selection_increment : float
        Lower feature importance threshold for increment feature selection.
    verbose : int
        Verbosity for logger.
    **extractor_kwargs : typing.Any
        Additional keyword arguments for feature extractor.
        Currently, only the `TemporalFeatureExtractor` module supports this parameter.
    """

    def __init__(
        self,
        target: str = "",
        mode: str = "",
        use_wavelets: bool = True,
        is_temporal: bool | None = None,
        extract_features: bool = True,
        collinear_threshold: float = 0.99,
        analyse_feature_sets: str = "auto",
        selection_cutoff: float = 0.85,
        selection_increment: float = 0.005,
        verbose: int = 1,
        **extractor_kwargs,
    ):
        BaseTransformer.__init__(self)
        LoggingMixin.__init__(self, verbose=verbose)
        self.target = target
        self.mode = mode

        check_dtypes(
            ("is_temporal", is_temporal, (bool, type(None))),
            ("extract_features", extract_features, bool),
            ("collinear_threshold", collinear_threshold, float),
            ("analyse_feature_sets", analyse_feature_sets, (str, type(None))),
            ("selection_cutoff", selection_cutoff, float),
            ("selection_increment", selection_increment, float),
        )
        for value, name in (
            (collinear_threshold, "collinear_threshold"),
            (selection_cutoff, "selection_cutoff"),
            (selection_increment, "selection_increment"),
        ):
            if not 0 < value < 1:
                raise ValueError(f"Invalid argument {name} = {value} ∉ (0, 1).")

        # Set attributes
        self.feature_extractor: BaseFeatureExtractor
        self.feature_selector = FeatureSelector(
            target, mode, selection_cutoff, selection_increment
        )
        self.is_temporal = is_temporal
        self.use_wavelets = use_wavelets
        self.extract_features = extract_features
        self.collinear_threshold = collinear_threshold
        self.analyse_feature_sets = analyse_feature_sets
        self.selection_cutoff = selection_cutoff
        self.selection_increment = selection_increment
        self.extractor_kwargs = extractor_kwargs
        self.collinear_cols_: list[str] = []

    def fit(self, data: pd.DataFrame, **fit_params):
        """Fits this feature processor (extractor & selector)

        Note: We implement fit_transform because we anyhow transform the data. Therefore,
            when using fit_transform we don't have to do redundant transformations.
        """
        self.fit_transform(data)
        return self

    def fit_transform(
        self, data: pd.DataFrame, feature_set: str | None = None, **fit_params
    ) -> pd.DataFrame:
        """Fits and transforms this feature processor."""
        self.logger.info("Fitting data.")
        check_data(data)

        # Remove collinear columns
        data = self._remove_collinear(data)

        # Fit and transform feature extractor.
        self._set_feature_extractor(data)
        data = self.feature_extractor.fit_transform(data)

        # Analyse feature importance and feature setssdfg
        data = self.feature_selector.fit_transform(data, feature_set=feature_set)
        self.feature_extractor.set_features(self.features_)

        self.is_fitted_ = True
        return data

    def transform(
        self, data: pd.DataFrame, feature_set: str | None = None
    ) -> pd.DataFrame:
        """
        Transform data and return it.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self.is_fitted_

        Parameters
        ----------
        data : pd.DataFrame
        feature_set : str, optional
            Desired feature set.
            When feature_set is None, all features will be returned.

        Returns
        -------
        pandas.DataFrame
        """
        check_data(data)
        self.logger.info("Transforming data.")
        if not self.is_fitted_:
            raise NotFittedError
        data = self._impute_missing_columns(data)

        # Set features for transformation
        if feature_set and feature_set in self.feature_sets_:
            self.set_feature_set(feature_set)
        elif feature_set:
            raise ValueError(f"Feature set does not exist: {feature_set}")

        # Transform
        data = self.feature_extractor.transform(data)
        return self.feature_selector.transform(data)

    def _set_feature_extractor(self, data: pd.DataFrame):
        """
        Checks is_temporal attribute. If not set and x is multi-indexed, sets to true.

        Parameters
        ----------
        x : pd.DataFrame
        """
        self.logger.debug("Checking whether to data has multi-index.")

        # Set is_temporal
        if self.is_temporal is None:
            self.is_temporal = len(data.index.names) == 2

        # Set feature extractor
        if not self.extract_features:
            self.feature_extractor = NopFeatureExtractor(
                target=self.target, mode=self.mode, verbose=self.verbose
            )
        elif self.is_temporal:
            self.feature_extractor = TemporalFeatureExtractor(
                target=self.target,
                mode=self.mode,
                fit_wavelets=self.use_wavelets,
                verbose=self.verbose,
                **self.extractor_kwargs,
            )
        else:
            self.feature_extractor = StaticFeatureExtractor(
                target=self.target,
                mode=self.mode,
                verbose=self.verbose,
            )

    def _remove_collinear(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Examines the data and separates different column types.

        Fitted attributes:
            Datetime columns are stored in "datetime_cols_".
            Collinear, numeric columns are stored in "collinear_cols_".
            Numeric columns (not collinear) are stored in "numeric_cols_".

        Parameters
        ----------
        data : pd.DataFrame
            Data to examine.
        """
        self.logger.info("Analysing columns of interest.")
        self.collinear_cols_ = find_collinear_columns(data, self.collinear_threshold)

        self.logger.info(f"Removed {len(self.collinear_cols_)} columns.")
        return data.drop(self.collinear_cols_, axis=1)

    def _impute_missing_columns(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing columns when not present for transforming.

        Parameters
        ----------
        x : pd.DataFrame
            Data to check and impute when necessary.

        Returns
        -------
        pd.DataFrame
            Cleaned data.
        """
        if not self.is_fitted_:
            raise NotFittedError

        # Find missing columns
        required_cols = [
            col
            for columns in translate_features(self.features_).values()
            for col in columns
        ]
        required_cols = list(set(required_cols))
        for col in [col for col in required_cols if col not in x]:
            warn(f"Imputing missing column: {col}.")
            x[col] = 0

        return x

    @property
    def features_(self) -> list[str]:
        """Returns extracted & selected features"""
        return self.feature_selector.features_

    @property
    def feature_importance_(self) -> dict[str, dict[str, float]]:
        """
        Format:
        {
            "rf": {
                "feature_1": 0.98,
                ...
            },
            ...
        }
        """
        return self.feature_selector.feature_importance_

    @property
    def feature_set_(self) -> str | None:
        return self.feature_selector.feature_set

    @property
    def feature_sets_(self) -> dict[str, list[str]]:
        """
        Format:
        {
            "rf": ["feature_1", ...],
            "rfi": ["feature_2", ...]
        }
        """
        return self.feature_selector.feature_sets_

    def set_feature_set(self, feature_set: str) -> None:
        """Updates the feature set of the feature selector & extractor"""
        self.feature_selector.feature_set = feature_set
        if self.feature_extractor:
            self.feature_extractor.set_features(self.features_)
