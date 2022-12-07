#  Copyright (c) 2022 by Amplo.

"""
Implements the basic behavior of feature processing.
"""


from __future__ import annotations

from abc import ABCMeta
from copy import deepcopy
from typing import Any, cast
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from amplo.base import BaseTransformer, LoggingMixin
from amplo.utils import check_dtypes, clean_column_names

__all__ = [
    "sanitize_series",
    "sanitize_dataframe",
    "BaseFeatureProcessor",
    "BaseFeatureExtractor",
]


def sanitize_series(series: pd.Series) -> pd.Series:
    """
    Sanitizes series data.

    - Replaces `np.inf` and `np.nan` with zeros.
    - Clips min and max values.
    - Converts all floats and ints to `np.float32` and `np.int32`, respectively.

    Parameters
    ----------
    series : pd.Series
        Data to be sanitized.

    Returns
    -------
    pd.Series
        Sanitized data.
    """
    dtype = series.dtype
    dtype = cast(npt.DTypeLike, dtype)  # type hint

    if pd.api.types.is_datetime64_any_dtype(series) or not np.issubdtype(
        dtype, np.number
    ):
        return series

    # Get info for 32-bit data
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(np.float32)
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(np.int32)
    else:
        raise TypeError(f"Invalid dtype for series data: {series.dtype}")
    # Sanitize
    lower = max(info.min, -1e12)  # type: ignore
    upper = min(info.max, 1e12)  # type: ignore
    series = series.fillna(0).clip(lower, upper).astype(info.dtype)
    return series


def sanitize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes each column.

    Parameters
    ----------
    frame : pd.DataFrame
        Data to be sanitized.

    Returns
    -------
    pd.DataFrame
        Sanitized data.
    """
    return frame.apply(sanitize_series, axis=0)


class BaseFeatureProcessor(BaseTransformer, LoggingMixin, metaclass=ABCMeta):
    """
    Base class for feature processors.

    Parameters
    ----------
    target : str, default: "target"
        Target column that must be present in data.
    mode : str, optional, default: None
        Model mode: {"classification", "regression", None}.
    verbose : int
        Verbosity for logger.
    """

    def __init__(self, target: str = "target", mode: str | None = None, verbose=0):
        BaseTransformer.__init__(self)
        LoggingMixin.__init__(self, verbose=verbose)

        check_dtypes(("target", target, str), ("mode", mode, (str, type(None))))

        if isinstance(mode, str):
            mode = mode.lower()
        if mode == "notset" or mode is None:
            # Allows the class to be initialized without specifying the mode
            mode = "notset"
        elif mode == "multiclass":
            mode = "classification"
        elif mode not in ("classification", "regression"):
            raise ValueError("Invalid `mode` argument.")

        self.mode = mode
        self.target = target

    @staticmethod
    def _check_x(x: pd.DataFrame, copy=True, sanitize=True) -> pd.DataFrame:
        """
        Check and sanitize x data.

        Parameters
        ----------
        x : pd.DataFrame
        copy : bool
        sanitize : bool

        Returns
        -------
        pd.DataFrame
        """
        check_dtypes("x", x, pd.DataFrame)

        x = deepcopy(x) if copy else x  # make copy

        # Stringify names
        x, _ = clean_column_names(x)
        if [col for col in x.columns if "__" in col]:
            # important for `raw_features_` attribute to work properly
            raise ValueError("Column names cannot contain '__' (double underscores).")
        if any(x.columns.duplicated()):
            raise ValueError("Feature column names are not unique.")

        # Sanitize values
        x_checked = sanitize_dataframe(x) if sanitize else x
        return x_checked

    @staticmethod
    def _check_y(y: pd.Series, copy=True, sanitize=True) -> pd.Series:
        """
        Check and sanitize x data.

        Parameters
        ----------
        y : pd.Series
        copy : bool
        sanitize : bool

        Returns
        -------
        pd.Series
        """
        check_dtypes("y", y, pd.Series)

        y = deepcopy(y) if copy else y  # make copy
        y.name = str(y.name) if y.name else "target"  # stringify name

        # Sanitize values
        y_checked = sanitize_series(y) if sanitize else y
        return y_checked

    @staticmethod
    def _check_x_y(
        x: pd.DataFrame, y: pd.Series, copy=True, sanitize=True
    ) -> tuple[pd.DataFrame, pd.Series]:

        # Pre-check
        x = BaseFeatureProcessor._check_x(x=x, copy=copy, sanitize=sanitize)
        y = BaseFeatureProcessor._check_y(y=y, copy=copy, sanitize=sanitize)

        # Check integrity
        if len(x) != len(y):
            raise ValueError("Length of x and y data does not match.")
        if not all(x.index == y.index):
            warn(
                "Indices of x and y data do not match. Setting x index as default.",
                UserWarning,
            )
            y.index = x.index

        return x, y

    def _check_data(
        self, data: pd.DataFrame, copy=True, sanitize=True, require_y=True
    ) -> tuple[pd.DataFrame, pd.Series]:

        x = data.drop(self.target, axis=1, errors="ignore")
        if not require_y:
            x = self._check_x(x, copy=copy, sanitize=sanitize)
            y = pd.Series(index=x.index, dtype="float64", name=self.target)  # dummy
            return x, y
        else:
            y = data[self.target]
            return self._check_x_y(x, y, copy=copy, sanitize=sanitize)


class BaseFeatureExtractor(BaseFeatureProcessor, metaclass=ABCMeta):
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

    Class attributes
    ----------------
    _add_to_settings : list of str
        Attribute names to be included in settings.
    _feature_translation : list of (str, str or None, str or None)
        Instructions for `FeatureProcessor.translate_features()` on how to relate
        extracted features to their original, raw features.
    """

    _add_to_settings = ["features_", *BaseFeatureProcessor._add_to_settings]

    def __init__(self, target: str = "target", mode: str | None = None, verbose=0):
        super().__init__(target=target, mode=mode, verbose=verbose)

        self.features_: list[str] = []
        self._validation_model = None
        self._baseline_scores = None

    def set_features(self, features: Any[str]) -> None:
        """
        (Re-)set the features_ attribute.

        Parameters
        ----------
        features : typing.Iterable of str
        """
        # Check input
        if isinstance(features, str):
            features = [features]
        check_dtypes(("feature_item", x, str) for x in features)
        # Apply
        self.features_ = sorted(features)

    def add_features(self, features: Any[str]) -> None:
        """
        Add items to the features_ attribute.

        Parameters
        ----------
        features : typing.Iterable of str
        """
        # Check input
        if isinstance(features, str):
            features = [features]
        check_dtypes(("feature_item", x, str) for x in features)
        # Apply
        self.features_.extend(features)
        self.features_ = sorted(set(self.features_))

    def remove_features(self, features: Any[str]) -> None:
        """
        Remove items in the features_ attribute.

        Parameters
        ----------
        features : typing.Iterable of str
        """
        # Check input
        if isinstance(features, str):
            features = [features]
        check_dtypes(("feature_item", x, str) for x in features)
        # Check integrity
        if not set(features).issubset(self.features_):
            raise ValueError(
                f"Cannot remove features that are not existing: "
                f"{set(features) - set(self.features_)}"
            )
        # Apply
        self.features_ = [x for x in self.features_ if x not in features]

    def _set_validation_model(self) -> None:
        """
        Set the validation model for feature scoring.
        """
        assert self.mode in ("classification", "regression"), "Invalid mode."
        if self.mode == "classification":
            self._validation_model = DecisionTreeClassifier(
                max_depth=3,
                class_weight="balanced",
                random_state=19483,
            )
        else:
            self._validation_model = DecisionTreeRegressor(
                max_depth=3,
                random_state=19483,
            )

    def _calc_feature_scores(
        self, feature: pd.Series, y: pd.Series
    ) -> npt.NDArray[np.floating]:
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
        scores : array of float
            Feature score. In case of multiclass, a score per class.
        """
        # (Re-)fit validation model.
        #  Note that we do not make a train-test split. In this case, it makes sense as
        #  we only fit a shallow tree (max_depth=3). Because of that the model cannot
        #  really overfit.
        feature_df = feature.to_frame()
        assert self._validation_model
        self._validation_model.fit(feature_df, y)

        # Score
        if self.mode == "classification":
            classes = (
                self.classes_
                if hasattr(self, "classes_")
                else self._validation_model.classes_
            )
            if len(classes) > 2:
                scores = [
                    self._validation_model.score(feature_df, y, y == c)
                    if c in self._validation_model.classes_
                    else 0.0
                    for c in classes
                ]  # weighted score
            else:
                scores = [self._validation_model.score(feature_df, y)]  # average score

        elif self.mode == "regression":
            scores = [self._validation_model.score(feature_df, y)]  # average score

        else:
            raise AttributeError("Invalid mode.")

        return np.array(scores)

    def _init_feature_baseline_scores(self, x: pd.DataFrame, y: pd.Series):
        """
        Initializes the baseline score of the given features.

        Parameters
        ----------
        x : pd.DataFrame
            Feature data.
        y : pd.Series
            Target data.
        """
        baseline_scores = x.apply(self._calc_feature_scores, y=y, axis=0)  # type: ignore
        baseline_scores = cast(pd.DataFrame, baseline_scores)
        self._baseline_scores = baseline_scores.max(1).to_list()

        if self.mode == "classification":
            self.classes_ = sorted(y.unique())

        self.logger.debug(f"Initialized the baseline score to {self._baseline_scores}")

    def _update_feature_baseline_scores(self, scores: pd.DataFrame):
        """
        Update the baseline scores.

        Parameters
        ----------
        scores : pd.DataFrame
            Scores where each column contains the scores for the given feature.
        """
        scores = pd.concat([scores, pd.Series(self._baseline_scores)], axis=1)
        self._baseline_scores = list(scores.max(1).values)

    def accept_feature(self, scores: npt.NDArray[np.floating]):
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
        if self._baseline_scores is None:
            warn("No baseline score is set. Output will be false", UserWarning)
        return any(scores >= 0.95 * np.array(self._baseline_scores))

    def select_scores(
        self, scores: pd.DataFrame, best_n_per_class=50, update_baseline=False
    ) -> pd.DataFrame:
        """
        Scores and selects each feature column.

        Parameters
        ----------
        scores : pd.DataFrame
            Scores to be selected.
        best_n_per_class : int
            Limit for the number of accepted features per class (take the best).
        update_baseline : bool
            Whether to update the baseline scores.

        Returns
        -------
        pd.DataFrame
            Scores for accepted features.

        Notes
        -----
        For the scores dataframe, values represent the scores (per class) and column
        names the respective feature name.
        """
        check_dtypes("scores", scores, pd.DataFrame)

        if scores.shape[1] == 0:
            return scores

        # For each class, add take the best `best_n_per_class` features
        selection = set()
        for _, row in scores.iterrows():
            cls_selection = row.sort_values().iloc[-best_n_per_class:]
            selection = selection.union(cls_selection.index)
        selection = list(selection)

        # Accept features that are better than the baseline
        accept_mask = scores[selection].apply(self.accept_feature, axis=0)
        accepted_scores = scores[selection].loc[:, accept_mask]

        # Update baseline and return accepted feature names
        if update_baseline:
            self._update_feature_baseline_scores(accepted_scores)

        return accepted_scores
