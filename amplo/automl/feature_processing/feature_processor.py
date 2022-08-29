#  Copyright (c) 2022 by Amplo.

"""
Feature processor for extracting and selecting features.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import psutil
from shap import TreeExplainer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from amplo.automl.feature_processing._base import (
    BaseFeatureExtractor,
    BaseFeatureProcessor,
    sanitize_series,
)
from amplo.automl.feature_processing.nop_feature_extractor import NopFeatureExtractor
from amplo.automl.feature_processing.static_feature_extractor import (
    StaticFeatureExtractor,
)
from amplo.automl.feature_processing.temporal_feature_extractor import (
    TemporalFeatureExtractor,
)
from amplo.base.exceptions import NotFittedError
from amplo.classification import CatBoostClassifier
from amplo.regression import CatBoostRegressor
from amplo.utils import check_dtypes

__all__ = ["find_collinear_columns", "FeatureProcessor"]


def find_collinear_columns(data, information_threshold=0.9):
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
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Invalid dtype for `data`: {type(data).__name__}")
    if not isinstance(information_threshold, float):
        raise ValueError(
            f"Invalid dtype for `threshold`: {type(information_threshold).__name__}"
        )
    if not 0 < information_threshold < 1:
        raise ValueError(
            "`information_threshold` must be a valid percentage, "
            "excluding 0 and 1, i.e., (0, 1)."
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
                corr_mat[i, j] = abs(sum_ / (ss_i * ss_j))

    # Set collinear columns
    mask = np.sum(corr_mat > information_threshold, axis=0) > 0
    collinear_columns = np.array(data.columns)[mask].astype(str).tolist()
    return collinear_columns


class FeatureProcessor(BaseFeatureProcessor):
    """
    Feature processor module to extract and select features.

    Parameters
    ----------
    mode : {"notset", "classification", "regression"}, optional, default: "notset"
        Model mode.
    is_temporal : bool, optional
        Whether the data should be treated as temporal data or not.
        If none is provided, is_temporal will be set to true when fit data is
        multi-indexed, false otherwise.
    extract_features : bool
        Whether to extract features or just remove correlating columns.
    collinear_threshold : float
        Information threshold for collinear features.
    analyse_feature_sets : {"auto", "all", "gini", "shap"}, optional, default: "auto"
        Which feature sets to analyse.
        If None, no feature set will be analysed.
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

    _add_to_settings = [
        *BaseFeatureProcessor._add_to_settings,
        "datetime_cols_",
        "collinear_cols_",
        "numeric_cols_",
        "feature_extractor",
        "feature_importance_",
        "feature_sets_",
    ]

    def __init__(
        self,
        mode="notset",
        is_temporal=None,
        extract_features=True,
        collinear_threshold=0.99,
        analyse_feature_sets="auto",
        selection_cutoff=0.85,
        selection_increment=0.005,
        verbose=1,
        **extractor_kwargs,
    ):
        super().__init__(mode=mode, verbose=verbose)

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
        self.is_temporal = is_temporal
        self.extract_features = extract_features
        self.collinear_threshold = collinear_threshold
        self.analyse_feature_sets = analyse_feature_sets
        self.selection_cutoff = selection_cutoff
        self.selection_increment = selection_increment
        self.feature_extractor = None
        self.extractor_kwargs = extractor_kwargs

    def _fit_transform(self, x, y=None, **fit_params):
        self.logger.info("Start fitting data.")

        # Input checks
        x, y = self._check_x_y(x, y)

        # Define which columns are datetime, numeric and collinear.
        self._find_columns_of_interest(x)
        x = x[self.numeric_cols_]

        # Fit and transform feature extractor.
        self._check_is_temporal_attribute(x)
        x_ext = self._fit_transform_feature_extractor(x, y)
        y_ext = self.transform_target(y)

        # Analyse feature importance and feature sets
        self._analyse_feature_sets(x_ext, y_ext)

        self.logger.info("Finished fitting.")
        return x_ext

    def transform(self, x, y=None, feature_set=None):
        """
        Transform data and return it.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Feature data to transform.
        y : numpy.ndarray or pandas.Series
            Additional target data to transform.
        feature_set : str, optional
            Desired feature set.
            When feature_set is None, all features will be returned.

        Returns
        -------
        pandas.DataFrame
            Transformed version of x.
        """
        # Check whether is fitted
        self.check_is_fitted()

        # Set features for transformation
        if feature_set is None:
            features = self.features_
        elif feature_set in self.feature_sets_:
            features = self.feature_sets_[feature_set]
        else:
            raise ValueError(f"Feature set does not exist: {feature_set}")

        # Remove features for faster transforming.
        # We (temporarily) remove features from the feature extractor for speed-up.
        # Only `features_` will be extracted.
        orig_features_ = self.features_
        self.features_ = features

        # Transform data
        xt = self._transform(x, y)

        # Restore original features
        self.features_ = orig_features_

        return xt

    def _transform(self, x, y=None, feature_set=None):
        self.logger.info("Transforming data.")

        # Handle input
        x = self._check_x(x)
        x = self._impute_missing_columns(x)

        # Transform
        return self.feature_extractor.transform(x)

    def transform_target(self, y):
        """
        Transform target column (only if temporal data).

        Parameters
        ----------
        y : pd.Series

        Returns
        -------
        pd.Series
        """
        if isinstance(self.feature_extractor, TemporalFeatureExtractor):
            y = self.feature_extractor._fit_data_to_window_size(y)  # noqa
            y = self.feature_extractor._pool_target(y)  # noqa

        return y

    # ----------------------------------------------------------------------
    # Feature name translation

    def translate_features(self, feature_columns: list[str]) -> dict[str, list[str]]:
        """
        Translates (extracted) features and tells its underlying original feature.

        Parameters
        ----------
        feature_columns : list of str
            Feature columns to be translated.

        Returns
        -------
        dict of {str: list of str}
            Dictionary with `feature_columns` as keys and their underlying original
            features as values.
        """
        check_dtypes(("item", item, str) for item in feature_columns)

        translation = {}
        for feature in feature_columns:
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
            elif re.search("^dist__", feature):
                # k-Means clusters need all numeric columns
                t = self.numeric_cols_
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

    def get_required_columns(self, feature_set=None):
        """
        Returns all required data columns that are required.

        Parameters
        ----------
        feature_set : str, optional
            Desired feature set.
            When no feature_set is `None` all features will be returned.

        Returns
        -------
        list of str
            All required feature columns for given feature set.
        """
        # Set features for transformation
        if feature_set is None:
            features = self.features_
        elif feature_set in self.feature_sets_:
            features = self.feature_sets_[feature_set]
        else:
            raise ValueError(f"Feature set does not exist: {feature_set}")

        # A feature name always contains before and/or after two underscores.
        # Based on this assumption we use RegEx to find all features that are
        # required for the given feature set (of feature names).
        all_cols = set(self.datetime_cols_ + self.numeric_cols_)
        required_cols = sorted(
            set(
                [
                    col
                    for col in all_cols
                    if any(re.search(f"(__{col}|{col}__)", f) for f in features)
                ]
            )
        )

        return required_cols

    # ----------------------------------------------------------------------
    # Feature processing

    def _check_is_temporal_attribute(self, x):
        """
        Checks is_temporal attribute. If not set and x is multi-indexed, sets to true.

        Parameters
        ----------
        x : pd.DataFrame
        """
        # Check if `is_temporal` attribute is set
        if self.is_temporal is None:
            # When x is multi-indexed, we assume that we have temporal data.
            self.is_temporal = len(x.index.names) == 2

    def _find_columns_of_interest(self, x):
        """
        Examines the data and separates different column types.

        Fitted attributes:
            Datetime columns are stored in "datetime_cols_".
            Collinear, numeric columns are stored in "collinear_cols_".
            Numeric columns (not collinear) are stored in "numeric_cols_".

        Parameters
        ----------
        x : pd.DataFrame
            Data to examine.
        """
        self.logger.info("Analysing columns of interest.")

        self.datetime_cols_ = [
            col for col in x if pd.api.types.is_datetime64_any_dtype(x[col])
        ]
        non_datetime_cols = [col for col in x if col not in self.datetime_cols_]
        self.collinear_cols_ = find_collinear_columns(
            x.loc[:, non_datetime_cols], self.collinear_threshold
        )
        self.numeric_cols_ = [
            col for col in non_datetime_cols if col not in self.collinear_cols_
        ]

        self.logger.info(
            f"Found {len(self.datetime_cols_)} datetime columns and "
            f"{len(self.collinear_cols_) + len(self.numeric_cols_)} numeric "
            f"columns whereas {len(self.collinear_cols_)} are collinear and "
            f"thus removed."
        )

    def _impute_missing_columns(self, x):
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
        self.check_is_fitted()

        # Find missing columns
        required_cols = [
            col
            for columns in self.translate_features(self.features_).values()
            for col in columns
        ]
        required_cols = list(set(required_cols))
        missing_cols = [col for col in required_cols if col not in x]
        missing_datetime = [col for col in missing_cols if col in self.datetime_cols_]
        missing_numeric = [col for col in missing_cols if col in self.numeric_cols_]
        assert set(missing_cols) == set(missing_datetime) | set(missing_numeric), (
            "Internal problem: "
            "Missing datetime and numeric columns don't add up to all missing columns."
        )
        if missing_datetime or missing_numeric:
            self.logger.warning(f"Imputing {len(missing_cols)} missing column(s).")

        # Impute missing datetime columns
        if missing_datetime:
            msg = "Imputing missing datetime columns is currently not supported."
            raise NotImplementedError(msg)

        # Impute missing numeric columns with zeros
        for col in missing_numeric:
            x[col] = sanitize_series(pd.Series([0] * x.shape[0]))

        return x

    def _fit_transform_feature_extractor(self, x, y):
        """
        Selects feature processor instance and calls fit_transform.

        Parameters
        ----------
        x : pd.DataFrame
        y : pd.Series

        Returns
        -------
        pd.DataFrame
            Extracted features.
        """
        assert self.is_temporal is not None, "Forgot to set attribute."

        # Fit feature extractor
        if not self.extract_features:
            self.feature_extractor = NopFeatureExtractor(mode=None)
        elif self.is_temporal:
            self.feature_extractor = TemporalFeatureExtractor(
                mode=self.mode, verbose=self.verbose, **self.extractor_kwargs
            )
        else:
            self.feature_extractor = StaticFeatureExtractor(
                mode=self.mode, verbose=self.verbose  # , **self.extractor_kwargs
            )

        # Extract features
        x_ext = self.feature_extractor.fit_transform(x, y)

        # Find collinear features and remove those
        collinear_cols = find_collinear_columns(x_ext, self.collinear_threshold)
        self.feature_extractor.remove_features(collinear_cols)
        self.logger.info(
            f"Removed {len(collinear_cols)} collinear columns "
            f"from extracted features."
        )

        return x_ext[self.feature_extractor.features_]

    @property
    def features_(self):
        """
        Returns `features_` attribute of the feature extractor.

        Raises
        ------
        NotFittedError
            If the feature extractor is not yet set.
        """
        if not isinstance(self.feature_extractor, BaseFeatureExtractor):
            raise NotFittedError("Invalid feature extractor.")
        return self.feature_extractor.features_

    @features_.setter
    def features_(self, value):
        """
        Setter for `features_` attribute.

        Parameters
        ----------
        value : typing.List of str

        Raises
        ------
        NotFittedError
            If the feature extractor is not yet set.
        """
        if not isinstance(self.feature_extractor, BaseFeatureExtractor):
            raise NotFittedError("Invalid feature extractor.")
        self.feature_extractor.set_features(value)

    # ----------------------------------------------------------------------
    # Feature selection

    def _analyse_feature_sets(self, x, y):
        """
        Explores importance of features and defines selected subsets.

        Parameters
        ----------
        x : pd.DataFrame
        y : pd.Series

        Notes
        -----
        When ``analyse_feature_sets`` is false, ``feature_importance_`` is not set and
        ``feature_sets_`` has only one set, containing all columns.
        """
        # Init
        self.feature_importance_ = {}
        self.feature_sets_ = {}

        # Analyse
        analyse_gini = self.analyse_feature_sets in ("auto", "all", "gini")
        analyse_shap = self.analyse_feature_sets in ("all", "shap") or (
            self.analyse_feature_sets == "auto" and len(y) < 50_000
        )
        if analyse_gini:
            self._select_gini_impurity(x, y)
        if analyse_shap:
            self._select_shap(x, y)
        if not (analyse_gini or analyse_shap):
            self.feature_sets_["take_all"] = list(x.columns)

        # Enforce feature sets being sorted
        self.feature_sets_ = {
            key: sorted(values) for key, values in self.feature_sets_.items()
        }

    def _select_gini_impurity(self, x, y):
        """
        Selects features based on the random forest feature importance.

        Calculates the mean decrease in Gini impurity. Symmetric correlation
        based on multiple features and multiple tree ensembles.

        Parameters
        ----------
        x : pd.DataFrame
        y : pd.Series
        """
        self.logger.info("Analysing feature importance: Gini impurity.")

        # Set model
        rs = np.random.RandomState(seed=236868)
        if self.mode == "regression":
            forest = RandomForestRegressor(random_state=rs)
        elif self.mode == "classification" or self.mode == "multiclass":
            forest = RandomForestClassifier(random_state=rs)
        else:
            raise ValueError("Invalid mode.")
        forest.fit(x, y)

        # Get RF values
        fi = forest.feature_importances_
        fi_sum = np.sum(fi)
        idx_sort = np.flip(np.argsort(fi))

        # Set feature importance
        self.feature_importance_["rf"] = {
            col: importance
            for col, importance in zip(x.columns[idx_sort], fi[idx_sort])
        }

        # Threshold: Take best n columns to satisfy selection cutoff
        mask = fi[idx_sort].cumsum() - fi[idx_sort] <= fi_sum * self.selection_cutoff
        idx_keep = idx_sort[mask]
        threshold = x.columns[idx_keep].to_list()
        self.feature_sets_["rf_threshold"] = threshold
        self.logger.info(
            f"Selected {len(threshold)} features with "
            f"{self.selection_cutoff * 100:.2f}% RF threshold."
        )

        # Increment
        idx_keep = fi > fi_sum * self.selection_increment
        increment = x.columns[idx_keep].to_list()
        self.feature_sets_["rf_increment"] = increment
        self.logger.info(
            f"Selected {len(increment)} features with "
            f"{self.selection_increment * 100:.2f}% RF increment."
        )

    def _select_shap(self, x, y):
        """
        Calculates shapely value to be used as a measure of feature importance.

        Parameters
        ----------
        x : pd.DataFrame
        y : pd.Series
        """
        self.logger.info("Analysing feature importance: Shapely additive explanations.")

        # Set model
        seed = 236868
        if self.mode == "regression":
            base = CatBoostRegressor(random_seed=seed)
        elif self.mode == "classification" or self.mode == "multiclass":
            base = CatBoostClassifier(random_seed=seed)
        else:
            raise ValueError("Invalid mode.")
        base.fit(x, y)

        # Get Shap values
        explainer = TreeExplainer(base.model)
        shap = np.array(explainer.shap_values(x, y))

        # Average over classes if necessary
        if shap.ndim == 3:
            shap = np.mean(np.abs(shap), axis=0)

        # Average over samples
        shap = np.mean(np.abs(shap), axis=0)
        shap /= shap.sum()  # normalize
        shap_sum = np.sum(shap)
        idx_sort = np.flip(np.argsort(shap))

        # Add to class attribute
        self.feature_importance_["shap"] = {
            col: importance
            for col, importance in zip(x.columns[idx_sort], shap[idx_sort])
        }

        # Threshold
        mask = (
            shap[idx_sort].cumsum() - shap[idx_sort] <= shap_sum * self.selection_cutoff
        )
        idx_keep = idx_sort[mask]
        threshold = x.columns[idx_keep].to_list()
        self.feature_sets_["shap_threshold"] = threshold
        self.logger.info(
            f"Selected {len(threshold)} features with "
            f"{self.selection_cutoff * 100:.2f}% Shap threshold."
        )

        # Increment
        idx_keep = shap > shap_sum * self.selection_increment
        increment = x.columns[idx_keep].to_list()
        self.feature_sets_["shap_increment"] = increment
        self.logger.info(
            f"Selected {len(increment)} features with "
            f"{self.selection_increment * 100:.2f}% Shap increment."
        )
