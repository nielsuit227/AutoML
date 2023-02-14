#  Copyright (c) 2022 by Amplo.

"""
Implements the basic behavior of feature processing.
"""

from __future__ import annotations

from typing import Any, TypeVar
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from amplo.base import BaseTransformer, LoggingMixin
from amplo.utils.util import check_dtypes, unique_ordered_list

__all__ = [
    "check_data",
    "assert_multi_index",
    "BaseFeatureExtractor",
    "PERFECT_SCORE",
]

PandasType = TypeVar("PandasType", pd.Series, pd.DataFrame)


PERFECT_SCORE = -1e-3


def assert_multi_index(
    data: PandasType,
    allow_single_index: bool = False,
) -> tuple[PandasType, bool]:
    """Checks whether provided data has a multi-index, and adds if not.

    parameters
    ----------
    data : pd.DataFrame | pd.Series
    allow_single_index : bool, default = False

    returns
    -------
    data : pd.DataFrame
    was_multiindex : bool
    """
    n_index_cols = len(data.index.names)

    if n_index_cols == 1 and not allow_single_index:
        raise ValueError("Data must be multiindexed.")
    if n_index_cols == 1:
        data.index = pd.MultiIndex.from_product([[0], data.index])
        return data, False
    elif n_index_cols == 2:
        return data, True
    else:
        raise ValueError("Data is neither single- nor properly multiindexed.")


def check_data(
    data: pd.DataFrame,
    allow_double_underscore: bool = False,
) -> None:
    """
    Checks validatity of data.

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
    """
    # Various checks
    if any("__" in str(col) for col in data.columns) and not allow_double_underscore:
        raise ValueError("Column names cannot contain '__' (double underscores).")
    if any(data.columns.duplicated()):
        raise ValueError("Data column names are not unique.")
    if data.isna().any().any():
        raise ValueError("Data contains NaN.")
    if any(not pd.api.types.is_numeric_dtype(data[k]) for k in data.columns):
        raise ValueError("Data contains non-numeric data.")
    if data.max().max() > 1e12 or data.min().min() < -1e12:
        raise ValueError("Data contains extreme values.")


class BaseFeatureExtractor(BaseTransformer, LoggingMixin):
    """
    Base class for feature extractors.

    Fitted attributes:
        Extracted feature names are stored in "features_".

    Parameters
    ----------
    target : str, default: "target"
        Target column that must be present in data.
    mode : str, optional, default: None
        Model mode: {"classification", "regression", None}.
    verbose : int
        Verbosity for logger.
    """

    def __init__(self, target: str = "target", mode: str = "classification", verbose=0):
        BaseTransformer.__init__(self)
        LoggingMixin.__init__(self, verbose=verbose)
        self.target = target
        self.mode = mode

        self.features_: list[str] = []
        self._validation_model = self.get_validation_model()
        self._baseline_score: float = -np.inf
        self.skipped_: bool = False
        self.is_fitted_ = False

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

    def add_features(self, features: list[str] | str | pd.DataFrame) -> None:
        """
        Add items to the features_ attribute.

        Parameters
        ----------
        features : typing.Iterable of str
        """
        # Check input
        if isinstance(features, pd.DataFrame):
            self.features_.extend(features.keys().tolist())
        elif isinstance(features, str):
            self.features_.append(features)
        elif isinstance(features, list):
            self.features_.extend(features)
        else:
            raise NotImplementedError
        self.features_ = unique_ordered_list(self.features_)

    def remove_features(self, features: list[str] | str) -> None:
        """
        Remove items in the features_ attribute.

        Parameters
        ----------
        features : typing.Iterable of str
        """
        # Check input
        if isinstance(features, str):
            features = [features]
        for x in features:
            check_dtypes(("feature_item", x, str))
        # Check integrity
        if not set(features).issubset(self.features_):
            raise ValueError(
                f"Cannot remove features that are not existing: "
                f"{set(features) - set(self.features_)}"
            )
        # Apply
        self.features_ = [x for x in self.features_ if x not in features]

    def get_validation_model(self) -> DecisionTreeClassifier | DecisionTreeRegressor:
        """
        Set the validation model for feature scoring.
        """
        assert self.mode in ("classification", "regression"), "Invalid mode."
        if self.mode == "classification":
            return DecisionTreeClassifier(
                max_depth=3,
                class_weight="balanced",
                random_state=19483,
            )
        else:
            warn(
                "There are known scoring issues for the DecisionTreeRegressor, as it is"
                " inherently bad at extrapolation."
            )
            return DecisionTreeRegressor(
                max_depth=3,
                random_state=19483,
            )

    def calc_feature_score(self, feature: pd.Series, y: pd.Series) -> float:
        """
        Analyses and scores a feature.

        Parameters
        ----------
        feature : pd.Series
            Feature to be analysed.
        y : pd.Series
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
        assert self._validation_model

        # Score
        if self.mode == "classification":
            if y.nunique() > 2:
                warn("We're not scoring features per class.")
            return np.mean(
                cross_val_score(
                    self._validation_model,
                    feature.values.reshape((-1, 1)),
                    y.values.reshape((-1, 1)),
                    scoring="neg_log_loss",
                    cv=2,
                )
            )
        elif self.mode == "regression":
            return np.mean(
                cross_val_score(
                    self._validation_model,
                    feature.values.reshape((-1, 1)),
                    y.values.reshape((-1, 1)),
                    scoring="neg_mean_squared_error",
                    cv=2,
                )
            )
        raise AttributeError("Invalid mode.")

    def initialize_baseline(self, x: pd.DataFrame, y: pd.Series):
        """
        Initializes the baseline score of the given features.

        Parameters
        ----------
        x : pd.DataFrame
            Feature data.
        y : pd.Series
            Target data.
        """
        baseline_scores = x.apply(self.calc_feature_score, y=y, axis=0)
        self._baseline_score = baseline_scores.max().max()
        assert self._baseline_score is not None

        if self._baseline_score > -1e-3:
            self.logger.info(
                "Baseline score large enough to skip feature extraction: "
                f"{self._baseline_score}"
            )

        self.logger.debug(f"Initialized the baseline score to {self._baseline_score}")

    def update_baseline(self, scores: npt.NDArray[Any] | float) -> None:
        """
        Update the baseline scores.

        Parameters
        ----------
        scores : pd.DataFrame
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
            warn("No baseline score is set. Output will be false", UserWarning)

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

    def select_scores(self, scores: pd.Series, update_baseline=True) -> pd.Series:
        """
        Scores and selects each feature column.

        Parameters
        ----------
        scores : pd.DataFrame
            Scores to be selected.
        update_baseline : bool
            Whether to update the baseline scores.

        Returns
        -------
        pd.Series
            Scores for accepted features.

        Notes
        -----
        For the scores dataframe, values represent the scores (per class) and column
        names the respective feature name.
        """
        check_dtypes(("scores", scores, pd.Series))

        if len(scores) == 0:
            return scores

        accepted = scores[scores.apply(self.accept_feature)]

        if update_baseline:
            self._baseline_score = scores.max()

        return accepted
