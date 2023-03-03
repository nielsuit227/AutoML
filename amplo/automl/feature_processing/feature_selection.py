#  Copyright (c) 2022 by Amplo.

"""
Feature selector for selecting features.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from shap import TreeExplainer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from amplo.base import BaseTransformer, LoggingMixin
from amplo.base.exceptions import NotFittedError
from amplo.classification import CatBoostClassifier
from amplo.regression import CatBoostRegressor
from amplo.utils import check_dtypes


class FeatureSelector(BaseTransformer, LoggingMixin):
    """
    Class that given a dataset, analyses the feature importance and creates feature sets
    that only include informative features.

    This is done in two ways:
    - Using the mean decrease in gini impurity of a random forest
    - Using SHapely Additive exPlanations (SHAP values)

    Per method, two feature sets are created, increment and threshold.
    The increment takes every feature that contributes more than `selection_increment`
    The cutoff takes the n highest features that combined carry a `selection_cutoff`
    fraction of the total feature importance.

    Parameters
    ----------
    target : str
    mode : str
    selection_cutoff : float
    selection_increment : float
    feature_set : str
        Can be set for transformation. Does not limit the selection process.
    analyser_feature_sets : str, default='auto'
        Either 'auto', 'all', 'gini' or 'shap'. Note that shap can be slow for large
        sample sets.
    verbose : int
    """

    def __init__(
        self,
        target: str,
        mode: str,
        selection_cutoff: float = 0.85,
        selection_increment: float = 0.005,
        feature_set: str | None = None,
        analyse_feature_sets: str = "auto",
        verbose: int = 1,
    ) -> None:
        BaseTransformer.__init__(self)
        LoggingMixin.__init__(self, verbose=verbose)

        check_dtypes(
            ("target", target, str),
            ("mode", mode, str),
            ("selection_cutoff", selection_cutoff, float),
            ("selection_increment", selection_increment, float),
        )

        self.target = target
        self.mode = mode
        self.selection_cutoff = selection_cutoff
        self.selection_increment = selection_increment
        self.analyse_feature_sets = analyse_feature_sets

        self.feature_set: str | None = feature_set
        self.feature_sets_: dict[str, list[str]] = {}
        self.feature_importance_: dict[str, dict[str, float]] = {}

    def fit(self, data: pl.DataFrame, index_cols: list[str]):  # type: ignore[override]
        """Fits this feature selector.

        If feature set is provided, it only selects using the corresponding method.

        Parameters
        ----------
        data : pl.DataFrame
            Fitting data.
        index_cols : list[str]
            Column names of index.
        """
        self.logger.info("Fitting feature selector.")

        # Select x and y data
        x = data.drop([self.target, *index_cols])
        y = data[self.target]

        if self.feature_set:
            if "rf" in self.feature_set:
                self.select_gini_impurity(x, y)
            elif "shap" in self.feature_set:
                self.select_shap(x, y)
            else:
                raise ValueError("Unknown provided feature set")

        else:
            if self.analyse_feature_sets in ("auto", "all", "gini"):
                self.select_gini_impurity(x, y)
            if self.analyse_feature_sets in ("all", "shap") or (
                self.analyse_feature_sets == "auto" and len(data) < 50_000
            ):
                self.select_shap(x, y)

        self.is_fitted_ = True
        return self

    def transform(  # type: ignore[override]
        self, data: pl.DataFrame, index_cols: list[str], feature_set: str | None = None
    ) -> pl.DataFrame:
        """Transforms feature sets

        Parameters
        ----------
        data : pl.DataFrame
            Transforming data.
        index_cols : list[str]
            Column names of index.
        feature_set : str, optional
            When not provided, the union of all feature sets is returned.

        Returns
        -------
        pl.DataFrame
            Transformed data.
        """
        if not self.is_fitted_:
            raise NotFittedError

        # Update feature set
        if feature_set:
            self.feature_set = feature_set
        elif not self.feature_set:
            self.logger.warning("Feature set not set, returning all features.")

        # Features_ is given from feature_set, so we can directly return
        if self.target in data:
            return data[[*index_cols, *self.features_, self.target]]
        return data[[*index_cols, *self.features_]]

    def fit_transform(  # type: ignore[override]
        self, data: pl.DataFrame, index_cols: list[str], feature_set: str | None = None
    ) -> pl.DataFrame:
        """
        Fits and transforms.

        Parameters
        ----------
        data : pl.DataFrame
            Fitting and transforming data.
        index_cols : list[str]
            Column names of index.
        feature_set : str | None, optional
            When not provided, the union of all feature sets is returned., by default None

        Returns
        -------
        pl.DataFrame
            Transformed data.
        """
        return self.fit(data, index_cols).transform(data, index_cols, feature_set)

    def select_gini_impurity(self, x: pl.DataFrame, y: pl.Series) -> None:
        """
        Selects features based on the random forest feature importance.

        Calculates the mean decrease in Gini impurity. Symmetric correlation
        based on multiple features and multiple tree ensembles.

        Parameters
        ----------
        x : pl.DataFrame
        y : pl.Series
        """
        self.logger.info("Analysing feature importance: Gini impurity.")

        # Set model
        rs = np.random.RandomState(seed=236868)
        if self.mode == "regression":
            forest = RandomForestRegressor(random_state=rs)
        elif self.mode in ("classification", "multiclass"):
            forest = RandomForestClassifier(random_state=rs)
        else:
            raise ValueError("Invalid mode.")
        forest.fit(x.to_pandas(), y.to_pandas())

        # Get RF values
        fi = forest.feature_importances_
        fi /= fi.sum()

        # Convert to dict
        self.feature_importance_["rf"] = self.sort_dict(
            {k: v for k, v in zip(x.columns, fi)}
        )

        # Make feature sets
        self.make_threshold("rf")
        self.make_increment("rf")
        self.logger.info(
            f"Selected {len(self.feature_sets_['rf_threshold'])} features with "
            f"{self.selection_cutoff * 100:.2f}% RF treshold."
        )
        self.logger.info(
            f"Selected {len(self.feature_sets_['rf_increment'])} features with "
            f"{self.selection_increment * 100:.2f}% RF increment."
        )

    def select_shap(self, x: pl.DataFrame, y: pl.Series) -> None:
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
        base: CatBoostClassifier | CatBoostRegressor
        if self.mode == "regression":
            base = CatBoostRegressor(random_seed=seed)
        elif self.mode in ("classification", "multiclass"):
            base = CatBoostClassifier(random_seed=seed)
        else:
            raise ValueError("Invalid mode.")
        base.fit(x.to_pandas(), y.to_pandas())

        # Get Shap values
        explainer = TreeExplainer(base.model)
        shap = np.array(explainer.shap_values(x.to_pandas(), y.to_pandas()))

        # Average over classes and samples and normalize
        if shap.ndim == 3:
            shap = np.mean(np.abs(shap), axis=0)
        shap = np.mean(np.abs(shap), axis=0)
        shap /= shap.sum()  # normalize

        # Convert to dict
        self.feature_importance_["shap"] = self.sort_dict(
            {k: v for k, v in zip(x.columns, shap)}
        )

        # Make feature sets
        self.make_threshold("shap")
        self.make_increment("shap")
        self.logger.info(
            f"Selected {len(self.feature_sets_['shap_threshold'])} features with "
            f"{self.selection_cutoff * 100:.2f}% Shap treshold."
        )
        self.logger.info(
            f"Selected {len(self.feature_sets_['shap_increment'])} features with "
            f"{self.selection_increment * 100:.2f}% Shap increment."
        )

    @property
    def features_(self) -> list[str]:
        """Returns the features of the current feature set"""
        if self.feature_set is None:
            return self.all_features
        return self.feature_sets_[self.feature_set]

    @property
    def all_features(self) -> list[str]:
        """Returns the union of all feature sets"""
        return list({f for s in self.feature_sets_.values() for f in s})

    def sort_dict(self, dct: dict[str, float]) -> dict[str, float]:
        """Sorts a dictionary by ascending values."""
        return dict(
            sorted(
                dct.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def make_threshold(self, feature_set: str) -> None:
        """Creates a feature set based on total information conservation."""
        fi = self.feature_importance_[feature_set]
        vals = np.array(list(fi.values()))
        total_info = np.cumsum(vals) - vals
        self.feature_sets_[f"{feature_set}_threshold"] = [
            k
            for i, (k, v) in enumerate(fi.items())
            if total_info[i] <= self.selection_cutoff
        ]

    def make_increment(self, feature_set: str) -> None:
        """Creates a feature set based on individual information carriage."""
        self.feature_sets_[f"{feature_set}_increment"] = [
            k
            for k, v in self.feature_importance_[feature_set].items()
            if v > self.selection_increment
        ]
