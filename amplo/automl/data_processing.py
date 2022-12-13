#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from amplo.base import LoggingMixin
from amplo.utils.util import check_dtypes, clean_column_names, clean_feature_name

__all__ = ["clean_feature_name", "DataProcessor"]


class DataProcessor(LoggingMixin):
    def __init__(
        self,
        target: str | None = None,
        include_output: bool = True,
        drop_datetime: bool = False,
        drop_constants: bool = False,
        drop_duplicate_rows: bool = False,
        missing_values: str = "interpolate",
        outlier_removal: str = "clip",
        z_score_threshold: int = 4,
        version: int = 1,
        verbose: int = 1,
    ):
        """
        Preprocessing Class. Cleans a dataset into a workable format.
        Deals with Outliers, Missing Values, duplicate rows, data types (floats,
        categorical and dates), Not a Numbers, Infinities.

        Parameters
        ----------
        target : str
            Column name of target variable
        include_output : bool
            Whether to include output in the data
        drop_datetime : bool
            Whether to drop datetime columns
        drop_contstants : bool
            If False, does not remove constants
        drop_duplicate_rows : bool
            If False, does not remove constant columns
        missing_values : {"remove_rows", "remove_cols", "interpolate", "mean", "zero"}
            How to deal with missing values.
        outlier_removal : {"quantiles", "z-score", "clip", "none"}
            How to deal with outliers.
        z_score_threshold : int
            If outlier_removal="z-score", the threshold is adaptable
        version : int
            Versioning the output files
        verbose : int
            How much to print
        """
        super().__init__(verbose=verbose)
        # Type checks
        check_dtypes(
            ("target", target, (type(None), str)),
            # ignore "int_cols" and "float_cols" as they're deprecated and unused
            ("include_output", include_output, bool),
            ("drop_datetime", drop_datetime, bool),
            ("drop_constants", drop_constants, bool),
            ("drop_duplicate_rows", drop_duplicate_rows, bool),
            ("z_score_threshold", z_score_threshold, int),
            ("drop_duplicate_rows", drop_duplicate_rows, bool),
            ("version", version, int),
            ("verbose", verbose, int),
        )

        # Integrity checks
        mis_values_algo = ["remove_rows", "remove_cols", "interpolate", "mean", "zero"]
        if missing_values not in mis_values_algo:
            raise ValueError(
                f"Missing values algorithm not implemented, pick from {mis_values_algo}"
            )
        out_rem_algo = ["quantiles", "z-score", "clip", "none"]
        if outlier_removal not in out_rem_algo:
            raise ValueError(
                f"Outlier Removal algorithm not implemented, pick from {out_rem_algo}"
            )

        # Arguments
        self.version = version
        self.include_output = include_output
        self.drop_datetime = drop_datetime
        self.target = target

        # Algorithms
        self.missing_values = missing_values
        self.outlier_removal = outlier_removal
        self.z_score_threshold = z_score_threshold
        self.drop_constants = drop_constants
        self.drop_duplicate_rows = drop_duplicate_rows

        # Fitted Settings
        self.num_cols_ = []
        self.bool_cols_ = []
        self.cat_cols_ = []
        self.date_cols_ = []
        self.dummies_ = {}
        self.q1_ = None
        self.q3_ = None
        self.means_ = None
        self.stds_ = None
        self.label_encodings_ = []

        # Info for Documenting
        self.is_fitted_ = False
        self.removed_duplicate_rows_ = 0
        self.removed_duplicate_columns_ = 0
        self.removed_outliers_ = 0
        self.imputed_missing_values_ = 0
        self.removed_constant_columns_ = 0

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits this data cleaning module and returns the transformed data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        # Clean column names and apply renaming
        data, self.rename_dict_ = clean_column_names(data)
        if self.target is not None:
            self.target = self.rename_dict_.get(self.target, None)

        # Remove target
        if not self.include_output and self.target is not None and self.target in data:
            data = data.drop(self.target, axis=1)

        # Remove Duplicates
        data = self.remove_duplicates(data, rows=self.drop_duplicate_rows)

        # Infer data-types
        data = self.infer_data_types(data)

        # Convert data types
        data = self.convert_data_types(data, fit_categorical=True)

        # Remove outliers
        data = self.fit_remove_outliers(data)

        # Remove missing values
        data = self.remove_missing_values(data)

        # Remove Constants
        if self.drop_constants:
            data = self.remove_constants(data)

        # Encode or decode target
        data = self.encode_labels(data, fit=True)

        # Finish
        self.is_fitted_ = True

        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Function that takes existing settings (including dummies), and transforms new
        data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        if not self.is_fitted_:
            raise ValueError("Transform only available for fitted objects.")

        # Clean column names and apply renaming
        data = data.rename(columns=self.rename_dict_)

        # Impute columns
        data = self._impute_columns(data)

        # Convert data types
        data = self.convert_data_types(data, fit_categorical=False)

        # Remove outliers
        data = self.remove_outliers(data)

        # Remove duplicates
        data = self.remove_duplicates(data)

        # Remove missing values
        data = self.remove_missing_values(data)

        # Encode or drop target
        data = self.encode_labels(data, fit=False)

        return data

    def get_settings(self) -> dict:
        """
        Get settings to recreate fitted object.
        """
        assert self.is_fitted_, "Object not yet fitted."
        settings = {
            "include_output": self.include_output,
            "drop_datetime": self.drop_datetime,
            "drop_constants": self.drop_constants,
            "drop_duplicate_rows": self.drop_duplicate_rows,
            "missing_values": self.missing_values,
            "outlier_removal": self.outlier_removal,
            "z_score_threshold": self.z_score_threshold,
            "rename_dict_": self.rename_dict_,
            "num_cols_": self.num_cols_,
            "bool_cols_": self.bool_cols_,
            "date_cols_": self.date_cols_,
            "cat_cols_": self.cat_cols_,
            "label_encodings_": self.label_encodings_,
            "means_": (
                self.means_.to_json() if isinstance(self.means_, pd.Series) else None
            ),
            "stds_": (
                self.stds_.to_json() if isinstance(self.stds_, pd.Series) else None
            ),
            "q1_": self.q1_.to_json() if isinstance(self.q1_, pd.Series) else None,
            "q3_": self.q3_.to_json() if isinstance(self.q3_, pd.Series) else None,
            "dummies_": self.dummies_,
            "imputed_missing_values_": self.imputed_missing_values_,
            "removed_outliers_": self.removed_outliers_,
            "removed_constant_columns_": self.removed_constant_columns_,
            "removed_duplicate_rows_": self.removed_duplicate_rows_,
            "removed_duplicate_columns_": self.removed_duplicate_columns_,
        }
        return settings

    def load_settings(self, settings: dict):
        """
        Loads settings from dictionary and recreates a fitted object
        """
        self.rename_dict_ = settings.get("rename_dict_", {})
        self.num_cols_ = settings.get("num_cols_", settings.get("num_cols", []))
        self.bool_cols_ = settings.get("bool_cols_", settings.get("bool_cols", []))
        self.date_cols_ = settings.get("date_cols_", settings.get("date_cols", []))
        self.cat_cols_ = settings.get("cat_cols_", settings.get("cat_cols", []))
        self.label_encodings_ = settings.get("label_encodings_", [])
        self.include_output = settings.get("include_output", True)
        self.drop_datetime = settings.get("drop_datetime", False)
        self.drop_constants = settings.get("drop_constants", False)
        self.drop_duplicate_rows = settings.get("drop_duplicate_rows", False)
        self.missing_values = settings.get("missing_values", [])
        self.outlier_removal = settings.get("outlier_removal", [])
        self.z_score_threshold = settings.get("z_score_threshold", [])
        self.means_ = settings.get("means_", settings.get("_means", None))
        self.stds_ = settings.get("stds_", settings.get("_stds", None))
        self.q1_ = settings.get("q1_", settings.get("_q1", None))
        self.q3_ = settings.get("q3_", settings.get("_q3", None))
        for key in ["means_", "stds_", "q1_", "q3_"]:
            if getattr(self, key):
                setattr(self, key, pd.read_json(getattr(self, key), typ="series"))
        self.dummies_ = settings.get("dummies_", settings.get("dummies", {}))
        self.is_fitted_ = True
        return self

    def infer_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        In case no data types are provided, this function infers the most likely data
        types

        parameters
        ----------
        data : pd.DataFrame

        returns
        -------
        data : pd.DataFrame
        """
        # Iterate through keys
        for key in data.keys():

            # Skip target -- we don't want to convert it
            if key == self.target:
                continue

            # Remove NaN for feature identification
            f = data[key]
            f = f[~f.isna()].infer_objects()

            # Integer and Float
            if pd.api.types.is_integer_dtype(f) or pd.api.types.is_float_dtype(f):
                self.num_cols_.append(key)
                continue

            # Datetime
            elif pd.api.types.is_datetime64_any_dtype(f):
                self.date_cols_.append(key)
                continue

            # Booleans
            elif pd.api.types.is_bool_dtype(f):
                self.bool_cols_.append(key)
                continue

            # Strings / Objects
            elif pd.api.types.is_object_dtype(f):

                # Check numerical
                numeric = pd.to_numeric(f, errors="coerce", downcast="integer")
                if numeric.isna().sum() < 0.3 * len(f):
                    self.num_cols_.append(key)

                    # Update data and continue
                    data[key] = numeric
                    continue

                # Check date
                date = pd.to_datetime(
                    f.astype("str"),
                    errors="coerce",
                    infer_datetime_format=True,
                )
                if date.isna().sum() < 0.3 * len(f):
                    self.date_cols_.append(key)
                    continue

                # Check categorical variable
                if data[key].nunique() < max(10, len(data) // 4):
                    self.cat_cols_.append(key)
                    continue

            # Else not found
            warn(f"Couldn't identify feature: {key}")
        return data

    def convert_data_types(
        self, data: pd.DataFrame, fit_categorical: bool = True
    ) -> pd.DataFrame:
        """
        Cleans up the data types of all columns.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        fit_categorical : bool
            Whether to fit the categorical encoder

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        # Drop unused columns & datetime columns
        data = data.drop(
            [
                k
                for k in data.keys()
                if k
                not in self.num_cols_
                + self.date_cols_
                + self.bool_cols_
                + self.cat_cols_
                + [self.target]
            ],
            axis=1,
        )

        if self.date_cols_ and self.drop_datetime:
            warn(
                f"Data contains datetime columns but are removed: '{self.date_cols_}'",
                UserWarning,
            )
            data = data.drop(self.date_cols_, axis=1)

        # Or convert to datetime (before done only on subset)
        elif self.date_cols_:
            for key in self.date_cols_:
                data[key] = pd.to_datetime(
                    data[key], errors="coerce", infer_datetime_format=True
                )

        # Integer columns
        for key in self.bool_cols_:
            data[key] = data[key].fillna(False).astype(np.int64)

        # Float columns
        for key in self.num_cols_:
            data[key] = pd.to_numeric(data[key], errors="coerce", downcast="float")

        # Categorical columns
        if fit_categorical:
            data = self._fit_cat_cols(data)
        else:
            assert self.is_fitted_, (
                ".convert_data_types() was called with fit_categorical=False, while "
                "categorical encoder is not yet fitted."
            )
        data = self._transform_cat_cols(data)

        return data

    def _fit_cat_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encoding categorical variables always needs a scheme. This fits the scheme.

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        for key in self.cat_cols_:
            # Clean the categorical variables
            data[key] = data[key].apply(clean_feature_name)
            is_nan = data[key].isna().any()

            # Get dummies, convert & store
            dummies = pd.get_dummies(data[key], prefix=key, dummy_na=is_nan)
            self.dummies_[key] = dummies.keys().tolist()

        return data

    def _transform_cat_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts categorical variables according to fitted scheme.

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        for key in self.cat_cols_:
            # Clean column
            data[key] = data[key].apply(clean_feature_name)

            # Transform
            value = self.dummies_[key]
            dummies = [i[len(key) + 1 :] for i in value]
            str_values = data[key].astype("str").values
            data[value] = np.equal.outer(str_values, dummies).astype(np.int64)
            data = data.drop(key, axis=1)
        return data

    def remove_duplicates(self, data: pd.DataFrame, rows: bool = False) -> pd.DataFrame:
        """
        Removes duplicate columns and rows.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        rows : bool
            Whether to remove duplicate rows --> not desirable to maintain timelines

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        # Note down
        n_rows, n_columns = len(data), len(data.keys())

        # Remove Duplicates
        if rows:
            data = data.drop_duplicates()
        data = data.loc[:, ~data.columns.duplicated()]  # type: ignore

        # Note
        self.removed_duplicate_columns = n_columns - len(data.keys())
        self.removed_duplicate_rows = n_rows - len(data)

        return data

    def remove_constants(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes constant columns

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        columns = len(data.keys())

        # Remove Constants
        const_cols = [
            col for col in data if data[col].nunique() == 1 and col != self.target
        ]
        data = data.drop(columns=const_cols)

        # Note
        self.removed_constant_columns = columns - len(data.keys())

        return data

    def fit_remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits parameters necessary to remove outliers and removes them using remove_outliers method.

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        # With quantiles
        if self.outlier_removal == "quantiles":
            self.q1_ = data[self.num_cols_].quantile(0.25)
            self.q3_ = data[self.num_cols_].quantile(0.75)

        # By z-score
        elif self.outlier_removal == "z-score":
            self.means_ = data[self.num_cols_].mean(skipna=True, numeric_only=True)
            self.stds_ = data[self.num_cols_].std(skipna=True, numeric_only=True)
            self.stds_[self.stds_ == 0] = 1

        return self.remove_outliers(data)

    def remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes outliers

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        # With Quantiles
        if self.outlier_removal == "quantiles":
            self.removed_outliers = (
                (data[self.num_cols_] > self.q3_).sum().sum()
                + (data[self.num_cols_] < self.q1_).sum().sum()
            ).tolist()
            data[self.num_cols_] = data[self.num_cols_].mask(
                data[self.num_cols_] < self.q1_
            )
            data[self.num_cols_] = data[self.num_cols_].mask(
                data[self.num_cols_] > self.q3_
            )

        # With z-score
        elif self.outlier_removal == "z-score":
            z_score = abs((data[self.num_cols_] - self.means_) / self.stds_)
            self.removed_outliers = (
                (z_score > self.z_score_threshold).sum().sum().tolist()  # type: ignore
            )
            data[self.num_cols_] = data[self.num_cols_].mask(
                z_score > self.z_score_threshold  # type: ignore
            )

        # With clipping
        elif self.outlier_removal == "clip":
            self.removed_outliers = (
                (data[self.num_cols_] > 1e12).sum().sum()
                + (data[self.num_cols_] < -1e12).sum().sum()
            ).tolist()
            data[self.num_cols_] = data[self.num_cols_].clip(lower=-1e12, upper=1e12)
        return data

    def remove_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values (infinities and "not a number"s)

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        # Replace infinities
        data = data.replace([np.inf, -np.inf], np.nan)

        # Note
        self.imputed_missing_values = data[self.num_cols_].isna().sum().sum().tolist()

        # Removes all rows with missing values
        if self.missing_values == "remove_rows":
            data.dropna(axis=0, inplace=True)

        # Removes all columns with missing values
        elif self.missing_values == "remove_cols":
            data.dropna(axis=1, inplace=True)

        # Fills all missing values with zero
        elif self.missing_values == "zero":
            data = data.fillna(0)

        # Mean and Interpolate require more than 1 value, use zero if less
        elif self.missing_values in ("interpolate", "mean") and len(data) <= 1:
            data = data.fillna(0)

        # Linearly interpolates missing values
        elif self.missing_values == "interpolate":
            # Get all non-date_cols & interpolate
            non_date = np.setdiff1d(data.keys().to_list(), self.date_cols_)
            data[non_date] = data[non_date].interpolate(limit_direction="both")

            # Fill rest (date & more missing values cols)
            data = self._interpolate_dates(data)

        # Fill missing values with column mean
        elif self.missing_values == "mean":
            non_date = np.setdiff1d(data.keys().to_list(), self.date_cols_)
            data[non_date] = data[non_date].fillna(data.mean())

            # Fill dates
            data = self._interpolate_dates(data)

        return data

    def _interpolate_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Unfortunately pandas does not support this out of the box. PR was made, but
        closed pre-merged. https://github.com/pandas-dev/pandas/pull/21915

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        for key in self.date_cols_:
            if data[key].isna().any():
                unix = data[key].astype("int64")
                unix[unix < 0] = np.nan  # NaT are -9e10
                unix = unix.interpolate(method="bfill").interpolate("pad")
                data[key] = pd.to_datetime(unix, unit="ns")
        return data

    def encode_labels(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """En- or decodes target column of `data`

        Parameters
        ----------
        data : pd.DataFrame
            input data
        fit : bool
            Whether to (re)fit the label encoder

        Returns
        -------
        data : pd.DataFrame
            With the encoded labels
        """
        # Get labels and encode / decode
        if not self.target or self.target not in data:
            return data
        if not self.include_output:
            return data.drop(self.target, axis=1)

        # Split output
        labels = data[self.target]

        # Check whether it's classification
        if labels.dtype == object or labels.nunique() <= labels.size / 2:

            # Create encoder
            encoder = LabelEncoder()
            if fit is True:
                encoder.fit(labels)
                self.label_encodings_ = pd.Series(encoder.classes_).to_list()
            elif not self.label_encodings_:
                raise NotFittedError("Encoder it not yet fitted")
            else:
                encoder.fit(self.label_encodings_)

            # Encode
            data[self.target] = encoder.transform(labels)
            return data

        # It's probably a regression task, thus no encoding needed
        warn(UserWarning("Labels are probably for regression. No encoding happened..."))
        return data

    def decode_labels(self, data: np.ndarray) -> pd.Series:
        """Decode labels from numerical dtype to original value

        Parameters
        ----------
        data : np.ndarray
            Input data

        Returns
        -------
        data : pd.Series
            With labels encoded

        Raises
        ------
        NotFittedError
            When `except_not_fitted` is True and label encoder is not fitted
        """
        # Checks
        if len(self.label_encodings_) == 0:
            raise NotFittedError(
                "Encoder it not yet fitted. Try first calling `encode_target` "
                "to set an encoding"
            )

        # Create encoder
        encoder = LabelEncoder()
        encoder.classes_ = np.array(self.label_encodings_)

        # Decode
        return pd.Series(encoder.inverse_transform(data))  # type: ignore

    def _impute_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        *** For production ***
        If a dataset is missing certain columns, this function looks at all registered
        columns and fills them with
        zeros.

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        data : pd.DataFrame
        """
        # Impute
        keys = self.date_cols_ + self.num_cols_ + self.bool_cols_ + self.cat_cols_
        to_impute = [k for k in keys if k not in data]
        data[to_impute] = np.zeros((len(data), len(to_impute)))

        # Warn
        if len(to_impute) > 0:
            warn(f"Imputed {len(to_impute)} missing columns! {to_impute}")
        return data

    def prune_features(self, features: list):
        """
        For use with AutoML.Pipeline. We practically never use all features. Yet this
        processor imputes any missing features. This causes redundant operations,
        memory, and warnings. This function prunes the features to avoid that.

        parameters
        ----------
        features : list
            Required features (NOTE: include required features for extracted)
        """
        hash_features = dict([(k, 0) for k in features])
        self.date_cols_ = [f for f in self.date_cols_ if f in hash_features]
        self.num_cols_ = [f for f in self.num_cols_ if f in hash_features]
        self.bool_cols_ = [f for f in self.bool_cols_ if f in hash_features]
        self.cat_cols_ = [f for f in self.cat_cols_ if f in hash_features]
