#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

from copy import deepcopy
from typing import Any
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd

from amplo.base import BaseTransformer, LoggingMixin
from amplo.base.exceptions import NotFittedError


class Standardizer(BaseTransformer, LoggingMixin):
    def __init__(
        self, target: str | None = None, mode: str = "classification", verbose: int = 1
    ):
        BaseTransformer.__init__(self)
        LoggingMixin.__init__(self, verbose=verbose)

        if mode not in ["classification", "regression"]:
            raise ValueError(
                "Invalid mode. Pick from 'classification' and 'regression'"
            )
        self.mode = mode
        self.target = target
        self.cols_: list[str] | None = None
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None
        self.is_fitted_ = False

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        self.is_fitted_ = True
        return self.transform(deepcopy(data))

    def fit(self, data: pd.DataFrame):
        # Gather cols
        self.cols_ = data.select_dtypes(include=np.number).columns.tolist()

        # Remove target if classification
        if (
            self.target
            and self.cols_
            and self.target in self.cols_
            and self.mode == "classification"
        ):
            self.cols_.remove(self.target)

        # Set attrs
        self.means_ = data[self.cols_].mean(axis=0)
        self.stds_ = data[self.cols_].std(axis=0)
        self.stds_[abs(self.stds_) <= 1e-9] = 1

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise NotFittedError
        assert self.cols_ is not None
        if not all(k in data for k in self.cols_):
            warn(f"Missing keys in data: {[k for k in self.cols_ if k not in data]}")

        data[self.cols_] = (data[self.cols_] - self.means_) / self.stds_
        return data

    def reverse(self, data: npt.NDArray[Any], column: str | None = None) -> pd.Series:
        """
        Currently only used to reverse predict regression predictions. Hence the natural
        support for np.ndarrays.
        """
        if not self.is_fitted_:
            raise NotFittedError
        assert (
            self.cols_ is not None
            and self.stds_ is not None
            and self.means_ is not None
        )

        # If columns are provided...
        if column:
            if data.ndim != 1:
                raise ValueError(
                    "When column is specified, data should be 1 dimensional."
                )
            data = data * self.stds_[column] + self.means_[column]
            return pd.Series(data)

        # ...and if not
        if len(self.cols_) != len(data[0]):
            raise ValueError("Unequal columns and provided column strings.")
        data = data * self.stds_ + self.means_
        return pd.Series(data)
