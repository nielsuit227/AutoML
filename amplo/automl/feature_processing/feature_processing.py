#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting and selecting features.
"""

from __future__ import annotations

import re
from itertools import combinations
from warnings import warn

import pandas as pd
import polars as pl

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
from amplo.utils.data import pandas_to_polars, polars_to_pandas

__all__ = [
    "find_collinear_columns",
    "translate_features",
    "get_required_columns",
    "FeatureProcessor",
]


def find_collinear_columns(
    data: pl.DataFrame, information_threshold: float = 0.9
) -> list[str]:
    """
    Finds collinear features and returns them.

    Calculates the Pearson Correlation coefficient for all input features.
    Features that exceed the information threshold are considered linearly
    co-dependent, i.e. describable by: y = a * x + b. As these features add
    little to no information, they will be removed.

    Parameters
    ----------
    data : pl.DataFrame
        Data to search for collinear features.
    information_threshold : float
        Percentage value that defines the threshold for a ``collinear`` feature.

    Returns
    -------
    list of str
        List of collinear feature columns.
    """
    check_dtypes(
        ("data", data, pl.DataFrame),
        ("information_threshold", information_threshold, float),
    )

    # Set helpers
    SPLITTER = "<->"

    # Calculate correlation within columns
    data_demeaned = data.fill_nan(None).with_columns(pl.all() - pl.all().mean())
    ss = data_demeaned.with_columns(pl.all().pow(2).sum().pow(0.5))[0]
    correlation = data_demeaned.select(
        [
            ((pl.col(coli) * pl.col(colj)).sum() / (ss[coli] * ss[colj]))
            .abs()
            .alias(f"{coli}{SPLITTER}{colj}")
            # 'combinations' iterates through all combinations of the column names
            # without having twice the same column and independent of the order
            for coli, colj in combinations(data.columns, 2)
        ]
    )

    # Filter out every column which succeeds the information threshold
    # NOTE: 'column', 'column_0' and 'field_{i}' are default names by polars
    collinear_columns = (
        # convert the dataframe to a series (kind of)
        correlation.transpose(include_header=True, header_name="column")
        # filter by information threshold
        .filter(pl.col("column_0") > information_threshold)["column"]
        # extract the column name (apply split and take the right hand side)
        .str.split_exact(SPLITTER, 2).struct.field("field_1")
        # convert to list
        .to_list()
    )
    # Sort and remove potential duplicates
    collinear_columns_ = sorted(set(map(str, collinear_columns)))

    return collinear_columns_


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

    def fit(self, data: pd.DataFrame):
        """
        Fits this feature processor (extractor & selector).

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        """
        # NOTE: We anyhow have to transform the data. Therefore, when calling
        # 'fit_transform' we do no redundant transformations.
        self.fit_transform(data)
        return self

    def fit_transform(
        self, data: pd.DataFrame, feature_set: str | None = None
    ) -> pd.DataFrame:
        """
        Fits and transforms this feature processor.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        feature_set : str | None, optional
            Choose specific feature set, by default None

        Returns
        -------
        pd.DataFrame
            Transformed data
        """

        self.logger.info("Fitting data.")

        # Convert to polars
        self.logger.debug("Convert pandas data to polars.")
        pl_data, index_renaming = pandas_to_polars(data)
        index_cols = list(index_renaming)

        # Check
        check_data(pl_data)

        # Remove collinear columns
        pl_data = self._remove_collinear(pl_data, index_cols)

        # Fit and transform feature extractor.
        self._set_feature_extractor(index_cols)
        pl_data = self.feature_extractor.fit_transform(pl_data, index_cols)

        # Analyse feature importance and feature setssdfg
        pl_data = self.feature_selector.fit_transform(pl_data, index_cols, feature_set)
        self.feature_extractor.set_features(self.features_)

        # Convert back to pandas and restore index
        self.logger.debug("Convert polars data back do pandas.")
        data = polars_to_pandas(pl_data, index_renaming)

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

        self.logger.info("Transforming data.")
        if not self.is_fitted_:
            raise NotFittedError

        # Convert to polars
        self.logger.debug("Convert pandas data to polars.")
        pl_data, index_renaming = pandas_to_polars(data)
        index_cols = list(index_renaming)

        # Check
        check_data(pl_data)

        # Set features for transformation
        if feature_set and feature_set in self.feature_sets_:
            self.set_feature_set(feature_set)
        elif feature_set:
            raise ValueError(f"Feature set does not exist: {feature_set}")

        # Transform
        pl_data = self._impute_missing_columns(pl_data)
        pl_data = self.feature_extractor.transform(pl_data, index_cols)
        pl_data = self.feature_selector.transform(pl_data, index_cols)

        # Convert back to pandas and restore index
        self.logger.debug("Convert polars data back do pandas.")
        data = polars_to_pandas(pl_data, index_renaming)

        return data

    def _set_feature_extractor(self, index_cols: list[str]):
        """
        Checks is_temporal attribute. If not set and x is multi-indexed, sets to true.

        Parameters
        ----------
        index_cols : list[str]
            Column names of the indices.
        """
        self.logger.debug("Setting feature extractor...")

        # Set is_temporal
        if len(index_cols) not in (1, 2):
            self.logger.warning("Index is neither single- nor double-indexed.")
        if self.is_temporal is None:
            self.is_temporal = len(index_cols) == 2
            self.logger.debug(
                f"Data is {'single' if self.is_temporal else 'double'}-indexed. "
                f"Setting 'is_temporal' attribute to {self.is_temporal}."
            )

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

        self.logger.debug(f"Chose {type(self.feature_extractor).__name__}.")

    def _remove_collinear(
        self, data: pl.DataFrame, index_cols: list[str]
    ) -> pl.DataFrame:
        """
        Examines the data and separates different column types.

        Fitted attributes:
            Datetime columns are stored in "datetime_cols_".
            Collinear, numeric columns are stored in "collinear_cols_".
            Numeric columns (not collinear) are stored in "numeric_cols_".

        Parameters
        ----------
        data : pl.DataFrame
            Data to examine.
        index_cols : list[str]
            Column names is the index.
        """
        self.logger.info("Analysing columns of interest.")
        self.collinear_cols_ = find_collinear_columns(
            data.drop(index_cols), self.collinear_threshold
        )

        self.logger.info(f"Removed {len(self.collinear_cols_)} columns.")
        return data.drop(self.collinear_cols_)

    def _impute_missing_columns(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Imputes missing columns when not present for transforming.

        Parameters
        ----------
        data : pl.DataFrame
            Data to check and impute when necessary.

        Returns
        -------
        pl.DataFrame
            Imputed data.
        """
        if not self.is_fitted_:
            raise NotFittedError

        # Identify required columns
        required_cols = [
            col
            for columns in translate_features(self.features_).values()
            for col in columns
        ]
        required_cols = list(set(required_cols))

        # Find missing columns and impute
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            warn(
                f"Imputing {len(missing_cols)} missing columns, namely: {missing_cols}"
            )
            data = data.with_columns([pl.lit(0).alias(col) for col in missing_cols])

        return data

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
