#  Copyright (c) 2022 by Amplo.

"""
Implements the basic behavior of feature processing.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from typing_extensions import Self

from amplo.base import BaseTransformer, LoggingMixin
from amplo.utils.util import check_dtypes, unique_ordered_list

__all__ = [
    "assert_double_index",
    "check_data",
    "BaseFeatureExtractor",
    "PERFECT_SCORE",
]


PERFECT_SCORE = -1e-3


def assert_double_index(
    data: pl.DataFrame, index_cols: list[str], allow_single: bool = False
) -> tuple[pl.DataFrame, list[str], bool]:
    """
    Checks whether provided data has a valid double-index.

    Parameters
    ----------
    data : pl.DataFrame
        Data to be checked.
    index_cols : list[str]
        Index column names. Only ['index'] and ['log', 'index'] are accepted.
    allow_single : bool, optional
        Add an index level if True and data is single-indexed, by default False

    Returns
    -------
    data : pl.DataFrame
        Checked data. May have an additional column due to index insertion.
    index_cols : list[str]
        Index column names. May have additional element due to index insertion.
    was_double_index : bool
        Evaluates to True when the input data was double-indexed already.
    """

    # Init helpers
    LOG, INDEX = "log", "index"
    was_double_index = True

    # Check index size and act if necessary
    if index_cols == [INDEX]:
        was_double_index = False
        if not allow_single:
            raise ValueError(
                "Data must be double-indexed. Got a single-indexed instead."
            )

        # Add log column, raise if exists already
        if LOG in data.columns:
            raise RuntimeError(f"New index column '{LOG}' already exists.")
        index_cols = [LOG, INDEX]  # assure that new index is first level
        # NOTE: 'pl.lit(0)' creates a new, '0'-filled column
        data = data.with_column(pl.lit(0).cast(pl.Int8).alias(LOG))

    elif index_cols != [LOG, INDEX]:
        if allow_single:
            raise ValueError(
                f"Data must be single- or double-indexed by ['{INDEX}'] or "
                f"['{LOG}', '{INDEX}'], respectively. Got {index_cols} instead."
            )
        raise ValueError(
            f"Data must be double-indexed by ['{LOG}', '{INDEX}']. "
            f"Got {index_cols} instead."
        )

    # Assert that indices exist
    if not set(index_cols).issubset(data.columns):
        raise ValueError("Index columns are not present in data.")

    return data, index_cols, was_double_index


def check_data(data: pl.DataFrame, allow_double_underscore: bool = False) -> None:
    """
    Checks validatity of data.

    Parameters
    ----------
    data : pl.DataFrame

    Raises
    ------
    ValueError
    """
    check_dtypes(("data", data, pl.DataFrame))

    # Various checks
    if any("__" in str(col) for col in data.columns) and not allow_double_underscore:
        raise ValueError("Column names cannot contain '__' (double underscores).")
    if data.fill_nan(None).null_count().max(axis=1).max() > 0:  # type: ignore[operator]
        raise ValueError("Data contains NaN.")
    if not all(
        data[col].dtype
        in (
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        )
        for col in data.columns
    ):
        raise ValueError("Data contains non-numeric data.")
    if data.max(axis=1).max() > 1e12 or data.min(axis=1).min() < -1e12:  # type: ignore[operator]
        raise ValueError("Data contains extreme values.")


class BaseFeatureExtractor(BaseTransformer, LoggingMixin):
    """
    Base class for feature extractors.

    Fitted attributes:
        Extracted feature names are stored in "features_".

    Parameters
    ----------
    target : str | None, optional
        Target column that must be present in data, by default None
    mode : str | None, optional
        Model mode: {"classification", "regression"}, by default None
    verbose : int, optional
        Verbisity for logger, by default 0
    """

    def __init__(self, target: str | None = None, mode: str | None = None, verbose=0):
        BaseTransformer.__init__(self)
        LoggingMixin.__init__(self, verbose=verbose)
        self.target = target or ""
        self.mode = mode or ""

        self.features_: list[str] = []
        self._validation_model = self.get_validation_model()
        self._baseline_score: float = -np.inf
        self.skipped_: bool = False
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, data: pl.DataFrame, index_cols: list[str]) -> Self:  # type: ignore[override]
        ...

    @abstractmethod
    def transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        ...

    @abstractmethod
    def fit_transform(self, data: pl.DataFrame, index_cols: list[str]) -> pl.DataFrame:  # type: ignore[override]
        ...

    def set_params(self, **params):
        super().set_params(**params)
        self._validation_model = self.get_validation_model()
        return self

    def set_features(self, features: list[str] | str) -> None:
        """
        (Re-)set the features_ attribute.

        Parameters
        ----------
        features : typing.Iterable of str
        """
        # Check input
        if isinstance(features, str):
            features = [features]
        for x in features:
            check_dtypes(("feature_item", x, str))
        # Apply
        self.features_ = sorted(features)

    def add_features(self, features: list[str] | str | pl.DataFrame) -> None:
        """
        Add items to the features_ attribute.

        Parameters
        ----------
        features : typing.Iterable of str
        """
        # Check input
        if isinstance(features, str):
            self.features_.append(features)
        elif isinstance(features, list):
            self.features_.extend(features)
        elif isinstance(features, pl.DataFrame):
            self.features_.extend(features.columns)
        else:
            raise NotImplementedError
        self.features_ = unique_ordered_list(self.features_)

    def get_validation_model(self) -> DecisionTreeClassifier | DecisionTreeRegressor:
        """
        Get the validation model for feature scoring.
        """
        if self.mode == "classification":
            return DecisionTreeClassifier(
                max_depth=3,
                class_weight="balanced",
                random_state=19483,
            )
        elif self.mode == "regression":
            self.logger.warning(
                "There are known scoring issues for the DecisionTreeRegressor, as it is"
                " inherently bad at extrapolation."
            )
            return DecisionTreeRegressor(
                max_depth=3,
                random_state=19483,
            )
        else:
            raise AttributeError(f"Invalid mode: '{self.mode}'")

    def initialize_baseline(self, x: pl.DataFrame, y: pl.Series):
        """
        Initializes the baseline score of the given features.

        Parameters
        ----------
        x : pl.DataFrame
            Feature data.
        y : pl.Series
            Target data.
        """

        # Calculate feature score for each column, keep the max
        self.logger.debug("Calculating baseline score for each column.")
        baseline = float("-inf")
        for col in x.columns:
            col_score = self.calc_feature_score(x[col], y)
            baseline = max(baseline, col_score)
        self._baseline_score = baseline

        if self._baseline_score > -1e-3:
            self.logger.info(
                "Baseline score large enough to skip feature extraction: "
                f"{self._baseline_score}"
            )
        self.logger.debug(f"Initialized the baseline score to {self._baseline_score}")

    def calc_feature_score(self, feature: pl.Series, y: pl.Series) -> float:
        """
        Analyses and scores a feature.

        Parameters
        ----------
        feature : pl.Series
            Feature to be analysed.
        y : pl.Series
            Target data (for scoring).

        Returns
        -------
        score : float
            Feature score. In case of multiclass, a score per class.
        """
        # (Re-)fit validation model.
        #  Note that we do not make a train-test split. In this case, it makes sense as
        #  we only fit a shallow tree (max_depth=3). Because of that the model cannot
        #  really overfit.

        if self.mode == "classification":
            if len(y.unique()) > 2:
                self.logger.warning("We're not scoring features per class.")
            return np.mean(
                cross_val_score(
                    self._validation_model,
                    feature.to_numpy().reshape((-1, 1)),
                    y.to_numpy().reshape((-1, 1)),
                    scoring="neg_log_loss",
                    cv=2,
                )
            )

        elif self.mode == "regression":
            return np.mean(
                cross_val_score(
                    self._validation_model,
                    feature.to_numpy().reshape((-1, 1)),
                    y.to_numpy().reshape((-1, 1)),
                    scoring="neg_mean_squared_error",
                    cv=2,
                )
            )

        raise AttributeError("Invalid mode.")

    def update_baseline(self, scores: npt.NDArray[Any] | float) -> None:
        """
        Update the baseline scores.

        Parameters
        ----------
        scores : npt.NDArray[Any] | float
            Scores where each column contains the scores for the given feature.
        """
        if self._baseline_score is None:
            raise ValueError("Baseline not yet set.")

        if isinstance(scores, float):
            self._baseline_score = max(self._baseline_score, scores)
        else:
            self._baseline_score = np.max(self._baseline_score, np.max(scores))

    def accept_feature(self, scores: npt.NDArray[Any] | float) -> bool:
        """
        Decides whether to accept a new feature.

        Parameters
        ----------
        scores : array of float
            Scores for checking against baseline threshold.

        Returns
        -------
        bool
            Whether to accept the feature.
        """
        if self._baseline_score is None:
            self.logger.warning("No baseline score is set. Output will be false")

        # If score is within 1% of baseline, accept.
        # NOTE: these scores are negative (neg_log_loss & neg_mean_square_error)

        if isinstance(scores, float):
            return scores >= self.weight_scheduler * self._baseline_score
        return any(scores >= self.weight_scheduler * self._baseline_score)

    @property
    def weight_scheduler(self) -> float:
        """
        We want to be lenient with adding features in the beginning, and stricter
        in the end to avoid adding too many features.
        """
        CUTOFF = 50

        # If scores are negative
        if self._baseline_score < 0:
            if len(self.features_) >= CUTOFF:
                return 0.98
            return 2 - np.log(len(self.features_) + 1) / np.log(CUTOFF + 1)

        # And if scores are positive
        if len(self.features_) >= CUTOFF:
            return 1.02
        return np.log(len(self.features_) + 1) / np.log(CUTOFF + 1)

    def select_scores(
        self, scores: dict[str, float], update_baseline=True
    ) -> list[str]:
        """
        Scores and selects each feature column.

        Parameters
        ----------
        scores : dict[str, float]
            Scores to be selected.
        update_baseline : bool
            Whether to update the baseline scores.

        Returns
        -------
        list[str]
            Scores for accepted features.
        """
        check_dtypes(("scores", scores, dict))

        if len(scores) == 0:
            return []

        accepted = []
        for key, value in scores.items():
            if self.accept_feature(value):
                accepted.append(key)

        if update_baseline:
            self._baseline_score = max(scores.values())

        return accepted
