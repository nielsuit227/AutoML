#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from amplo.utils.util import check_dtypes, clean_column_names, clean_feature_name

__all__ = ["clean_feature_name", "DataProcessor"]


class DataProcessor:
    def __init__(
        self,
        target: str | None = None,
        num_cols: list[str] | None = None,
        bool_cols: list[str] | None = None,
        date_cols: list[str] | None = None,
        cat_cols: list[str] | None = None,
        int_cols: list[str] | None = None,  # deprecated
        float_cols: list[str] | None = None,  # deprecated
        include_output: bool = True,
        drop_datetime: bool = False,
        remove_constants: bool = True,
        missing_values: str = "interpolate",
        outlier_removal: str = "clip",
        z_score_threshold: int = 4,
        version: int = 1,
        verbosity: int = 1,
    ):
        """
        Preprocessing Class. Cleans a dataset into a workable format.
        Deals with Outliers, Missing Values, duplicate rows, data types (floats,
        categorical and dates), Not a Numbers, Infinities.

        Parameters
        ----------
        target : str
            Column name of target variable
        num_cols : list
            Float columns
        bool_cols : list
            Boolean columns
        date_cols : list
            Date columns, all parsed to pd.datetime format
        cat_cols : list
            Categorical Columns. Currently, all one-hot encoded.
        int_cols : list[str]
            Deprecated
        float_cols : list[str]
            Deprecated
        include_output : bool
            Whether to include output in the data
        drop_datetime : bool
            Whether to drop datetime columns
        remove_constants : bool
            If False, does not remove constant columns
        missing_values : {"remove_rows", "remove_cols", "interpolate", "mean", "zero"}
            How to deal with missing values.
        outlier_removal : {"quantiles", "z-score", "clip", "none"}
            How to deal with outliers.
        z_score_threshold : int
            If outlier_removal="z-score", the threshold is adaptable
        version : int
            Versioning the output files
        verbosity : int
            How much to print
        """

        # Type checks
        check_dtypes(
            ("target", target, (type(None), str)),
            ("num_cols", num_cols, (type(None), list)),
            ("bool_cols", bool_cols, (type(None), list)),
            ("date_cols", date_cols, (type(None), list)),
            ("cat_cols", cat_cols, (type(None), list)),
            # ignore "int_cols" and "float_cols" as they're deprecated and unused
            ("include_output", include_output, bool),
            # ignore "missing_values" and "outlier_removal" as they're checked below
            ("z_score_threshold", z_score_threshold, int),
            ("remove_constants", remove_constants, bool),
            ("version", version, int),
            ("verbosity", verbosity, int),
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
        self.target = target or None
        self.num_cols = num_cols or []
        self.bool_cols = bool_cols or []
        self.cat_cols = cat_cols or []
        self.date_cols = date_cols or []

        # Deprecation warnings
        if int_cols is not None:
            raise DeprecationWarning("int_cols is deprecated.")
        if float_cols is not None:
            raise DeprecationWarning("float_cols is deprecated.")

        # Algorithms
        self.missing_values = missing_values
        self.outlier_removal = outlier_removal
        self.z_score_threshold = z_score_threshold
        self.removeConstants = remove_constants

        # Fitted Settings
        self.data = None
        self.dummies = {}
        self._q1 = None
        self._q3 = None
        self._means = None
        self._stds = None
        self._label_encodings = []

        # Info for Documenting
        self.is_fitted = False
        self.verbosity = verbosity
        self.removed_duplicate_rows = 0
        self.removed_duplicate_columns = 0
        self.removed_outliers = 0
        self.imputed_missing_values = 0
        self.removed_constant_columns = 0

    def _fit_transform(
        self, data: pd.DataFrame, fit=False, remove_constants=True
    ) -> "DataProcessor":
        """
        Wraps behavior of both, fitting and transforming the DataProcessor.
        The function basically reduces duplicated code fragments of `self.fit_transform`
         and `self.transform`.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        fit : bool
            If True, it will fit the transformer, too
        remove_constants : bool
            If True, it will remove constants when fit

        Returns
        -------
        DataProcessor
        """

        # Clean column names and apply renaming
        if fit:
            self.data, self.rename_dict_ = clean_column_names(data)
            if self.target is not None:
                self.target = self.rename_dict_.get(self.target, None)
            self.num_cols = [self.rename_dict_[col] for col in self.num_cols]
            self.bool_cols = [self.rename_dict_[col] for col in self.bool_cols]
            self.cat_cols = [self.rename_dict_[col] for col in self.cat_cols]
            self.date_cols = [self.rename_dict_[col] for col in self.date_cols]
        else:
            self.data = data.rename(columns=self.rename_dict_)

        # Impute columns
        self._impute_columns()

        # Remove target
        if (
            fit
            and not self.include_output
            and self.target is not None
            and self.target in self.data
        ):
            self.data = self.data.drop(self.target, axis=1)

        if fit:
            # Infer data-types
            self.infer_data_types()

            # Remove Duplicates
            self.remove_duplicates(rows=not isinstance(self.data.index, pd.MultiIndex))

        # Convert data types
        self.convert_data_types(fit_categorical=fit)

        # Remove outliers
        self.remove_outliers(fit=fit)

        # Remove missing values
        self.remove_missing_values()

        # Remove Constants
        if fit and remove_constants:
            self.remove_constants()

        # Convert integer columns
        self.convert_float_int()

        # Encode or decode target
        self._code_target_column(fit=fit)

        return self

    def fit_transform(self, data: pd.DataFrame, remove_constants=True) -> pd.DataFrame:
        """
        Fits this data cleaning module and returns the transformed data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        remove_constants : bool
            If True, it will remove constants when fit

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """

        self._fit_transform(data, fit=True, remove_constants=remove_constants)
        assert isinstance(self.data, pd.DataFrame)

        # Finish
        self.is_fitted = True

        return self.data

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
        if not self.is_fitted:
            raise ValueError("Transform only available for fitted objects.")

        self._fit_transform(data, fit=False)
        assert isinstance(self.data, pd.DataFrame)

        return self.data

    def get_settings(self) -> dict:
        """
        Get settings to recreate fitted object.
        """
        assert self.is_fitted, "Object not yet fitted."
        settings = {
            "rename_dict_": self.rename_dict_,
            "num_cols": self.num_cols,
            "bool_cols": self.bool_cols,
            "date_cols": self.date_cols,
            "cat_cols": self.cat_cols,
            "_label_encodings": self._label_encodings,
            "missing_values": self.missing_values,
            "outlier_removal": self.outlier_removal,
            "z_score_threshold": self.z_score_threshold,
            "_means": (
                self._means.to_json() if isinstance(self._means, pd.Series) else None
            ),
            "_stds": (
                self._stds.to_json() if isinstance(self._stds, pd.Series) else None
            ),
            "_q1": self._q1.to_json() if isinstance(self._q1, pd.Series) else None,
            "_q3": self._q3.to_json() if isinstance(self._q3, pd.Series) else None,
            "dummies": self.dummies,
            "fit": {
                "imputed_missing_values": self.imputed_missing_values,
                "removed_outliers": self.removed_outliers,
                "removed_constant_columns": self.removed_constant_columns,
                "removed_duplicate_rows": self.removed_duplicate_rows,
                "removed_duplicate_columns": self.removed_duplicate_columns,
            },
        }
        return settings

    def load_settings(self, settings: dict) -> None:
        """
        Loads settings from dictionary and recreates a fitted object
        """
        self.rename_dict_ = settings.get("rename_dict_", {})
        self.num_cols = settings.get("num_cols", [])
        self.bool_cols = settings.get("bool_cols", [])
        self.date_cols = settings.get("date_cols", [])
        self.cat_cols = settings.get("cat_cols", [])
        self._label_encodings = settings.get("_label_encodings", [])
        self.missing_values = settings.get("missing_values", [])
        self.outlier_removal = settings.get("outlier_removal", [])
        self.z_score_threshold = settings.get("z_score_threshold", [])
        self._means = (
            None
            if settings["_means"] is None
            else pd.read_json(settings["_means"], typ="series")
        )
        self._stds = (
            None
            if settings["_stds"] is None
            else pd.read_json(settings["_stds"], typ="series")
        )
        self._q1 = (
            None
            if settings["_q1"] is None
            else pd.read_json(settings["_q1"], typ="series")
        )
        self._q3 = (
            None
            if settings["_q3"] is None
            else pd.read_json(settings["_q3"], typ="series")
        )
        self.dummies = settings.get("dummies", {})
        self.is_fitted = True

    def infer_data_types(self):
        """
        In case no data types are provided, this function infers the most likely data
        types
        """
        if len(self.cat_cols) == len(self.num_cols) == len(self.date_cols) == 0:
            assert isinstance(self.data, pd.DataFrame)

            # Iterate through keys
            for key in self.data.keys():

                # Skip target
                if key == self.target:
                    continue

                # Remove NaN for feature identification
                f = self.data[key]
                f = f[~f.isna()].infer_objects()

                # Integer and Float
                if pd.api.types.is_integer_dtype(f) or pd.api.types.is_float_dtype(f):
                    self.num_cols.append(key)
                    continue

                # Datetime
                elif pd.api.types.is_datetime64_any_dtype(f):
                    self.date_cols.append(key)
                    continue

                # Booleans
                elif pd.api.types.is_bool_dtype(f):
                    self.bool_cols.append(key)
                    continue

                # Strings / Objects
                elif pd.api.types.is_object_dtype(f):

                    # Check numerical
                    numeric = pd.to_numeric(f, errors="coerce", downcast="integer")
                    if numeric.isna().sum() < 0.3 * len(f):
                        self.num_cols.append(key)

                        # Update data and continue
                        self.data[key] = numeric
                        continue

                    # Check date
                    date = pd.to_datetime(
                        f.astype("str"),
                        errors="coerce",
                        infer_datetime_format=True,
                    )
                    if date.isna().sum() < 0.3 * len(f):
                        self.date_cols.append(key)
                        continue

                    # Check categorical variable
                    if self.data[key].nunique() < max(10, len(self.data) // 4):
                        self.cat_cols.append(key)
                        continue

                # Else not found
                warnings.warn(f"Couldn't identify feature: {key}")

    def convert_data_types(
        self, data=None, fit_categorical: bool = True
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
        # Set data
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            self.data = data
        assert isinstance(self.data, pd.DataFrame)

        # Drop Datetime columns
        if self.date_cols and self.drop_datetime:
            warnings.warn(
                f"Data contains datetime columns but are removed: '{self.date_cols}'",
                UserWarning,
            )
            self.data = self.data.drop(self.date_cols, axis=1)
        # Or convert to datetime (before done only on subset)
        elif self.date_cols:
            for key in self.date_cols:
                self.data[key] = pd.to_datetime(
                    self.data[key], errors="coerce", infer_datetime_format=True
                )

        # Integer columns
        for key in self.bool_cols:
            self.data[key] = self.data[key].astype(np.int64)

        # Float columns
        for key in self.num_cols:
            self.data[key] = pd.to_numeric(
                self.data[key], errors="coerce", downcast="float"
            )

        # Categorical columns
        if fit_categorical:
            self._fit_cat_cols()
        else:
            assert self.is_fitted, (
                ".convert_data_types() was called with fit_categorical=False, while "
                "categorical encoder is not yet fitted."
            )
        self.data = self._transform_cat_cols()

        return self.data

    def _fit_cat_cols(self, data=None):
        """
        Encoding categorical variables always needs a scheme. This fits the scheme.
        """
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            self.data = data
        assert isinstance(self.data, pd.DataFrame)

        for key in self.cat_cols:
            # Clean the categorical variables
            is_nan = self.data[key].isna().any()
            self.data[key] = self.data[key].apply(clean_feature_name)

            # Get dummies, convert & store
            dummies = pd.get_dummies(self.data[key], prefix=key, dummy_na=is_nan)
            self.dummies[key] = dummies.keys().tolist()

        return self

    def _transform_cat_cols(self, data=None) -> pd.DataFrame:
        """
        Converts categorical variables according to fitted scheme.
        """
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            self.data = data
        assert isinstance(self.data, pd.DataFrame)

        for key in self.cat_cols:
            # Clean column
            self.data[key] = self.data[key].apply(clean_feature_name)

            # Transform
            value = self.dummies[key]
            dummies = [i[len(key) + 1 :] for i in value]
            str_values = self.data[key].astype("str").values
            self.data[value] = np.equal.outer(str_values, dummies).astype(np.int64)  # type: ignore
            self.data = self.data.drop(key, axis=1)
        return self.data

    def remove_duplicates(self, data=None, rows: bool = True) -> pd.DataFrame:
        """
        Removes duplicate columns and rows.

        Parameters
        ----------
        rows : bool
            Whether to remove duplicate rows --> not desirable to maintain timelines
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            self.data = data
        assert isinstance(self.data, pd.DataFrame)

        # Note down
        n_rows, n_columns = len(self.data), len(self.data.keys())

        # Remove Duplicates
        if rows:
            self.data = self.data.drop_duplicates()
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]  # type: ignore

        # Note
        self.removed_duplicate_columns = n_columns - len(self.data.keys())
        self.removed_duplicate_rows = n_rows - len(self.data)

        assert isinstance(self.data, pd.DataFrame)
        return self.data

    def remove_constants(self, data=None) -> pd.DataFrame:
        """
        Removes constant columns
        """
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            self.data = data
        assert isinstance(self.data, pd.DataFrame)
        columns = len(self.data.keys())

        # Remove Constants
        if self.removeConstants:
            const_cols = [
                col
                for col in self.data
                if self.data[col].nunique() == 1 and col != self.target
            ]
            self.data = self.data.drop(columns=const_cols)

        # Note
        self.removed_constant_columns = columns - len(self.data.keys())

        return self.data

    def fit_outliers(self, data: pd.DataFrame | None = None):
        """
        Checks outliers
        """
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            self.data = data
        assert isinstance(self.data, pd.DataFrame)

        # With quantiles
        if self.outlier_removal == "quantiles":
            self._q1 = self.data[self.num_cols].quantile(0.25)
            self._q3 = self.data[self.num_cols].quantile(0.75)

        # By z-score
        elif self.outlier_removal == "z-score":
            self._means = self.data[self.num_cols].mean(skipna=True, numeric_only=True)
            self._stds = self.data[self.num_cols].std(skipna=True, numeric_only=True)
            self._stds[self._stds == 0] = 1

    def remove_outliers(
        self, data: pd.DataFrame | None = None, fit: bool = True
    ) -> pd.DataFrame:
        """
        Removes outliers
        """
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            self.data = data
        assert isinstance(self.data, pd.DataFrame)

        # Check if needs fitting
        if fit:
            self.fit_outliers(self.data)
        else:
            if not self.is_fitted:
                raise ValueError(
                    ".remove_outliers() is called with fit=False, yet the object isn't"
                    " fitted yet."
                )

        # With Quantiles
        if self.outlier_removal == "quantiles":
            self.removed_outliers = (
                (self.data[self.num_cols] > self._q3).sum().sum()
                + (self.data[self.num_cols] < self._q1).sum().sum()
            ).tolist()
            self.data[self.num_cols] = self.data[self.num_cols].mask(
                self.data[self.num_cols] < self._q1
            )
            self.data[self.num_cols] = self.data[self.num_cols].mask(
                self.data[self.num_cols] > self._q3
            )

        # With z-score
        elif self.outlier_removal == "z-score":
            z_score = abs((self.data[self.num_cols] - self._means) / self._stds)
            self.removed_outliers = (
                (z_score > self.z_score_threshold).sum().sum().tolist()  # type: ignore
            )
            self.data[self.num_cols] = self.data[self.num_cols].mask(
                z_score > self.z_score_threshold  # type: ignore
            )

        # With clipping
        elif self.outlier_removal == "clip":
            self.removed_outliers = (
                (self.data[self.num_cols] > 1e12).sum().sum()
                + (self.data[self.num_cols] < -1e12).sum().sum()
            ).tolist()
            self.data[self.num_cols] = self.data[self.num_cols].clip(
                lower=-1e12, upper=1e12
            )
        return self.data

    def remove_missing_values(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Fills missing values (infinities and "not a number"s)
        """
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            self.data = data
        assert isinstance(self.data, pd.DataFrame)

        # Replace infinities
        self.data = self.data.replace([np.inf, -np.inf], np.nan)

        # Note
        self.imputed_missing_values = (
            self.data[self.num_cols].isna().sum().sum().tolist()
        )

        # Removes all rows with missing values
        if self.missing_values == "remove_rows":
            self.data.dropna(axis=0, inplace=True)

        # Removes all columns with missing values
        elif self.missing_values == "remove_cols":
            self.data.dropna(axis=1, inplace=True)

        # Fills all missing values with zero
        elif self.missing_values == "zero":
            self.data = self.data.fillna(0)

        # Mean and Interpolate require more than 1 value, use zero if less
        elif self.missing_values in ("interpolate", "mean") and len(self.data) <= 1:
            self.data = self.data.fillna(0)

        # Linearly interpolates missing values
        elif self.missing_values == "interpolate":
            # Get all non-date_cols & interpolate
            non_date = np.setdiff1d(self.data.keys().to_list(), self.date_cols)
            self.data[non_date] = self.data[non_date].interpolate(
                limit_direction="both"
            )

            # Fill rest (date & more missing values cols)
            self._interpolate_dates()

        # Fill missing values with column mean
        elif self.missing_values == "mean":
            non_date = np.setdiff1d(self.data.keys().to_list(), self.date_cols)
            self.data[non_date] = self.data[non_date].fillna(self.data.mean())

            # Fill dates
            self._interpolate_dates()

        assert isinstance(self.data, pd.DataFrame)
        return self.data

    def _interpolate_dates(self):
        """
        Unfortunately pandas does not support this out of the box. PR was made, but
        closed pre-merged. https://github.com/pandas-dev/pandas/pull/21915
        """
        assert isinstance(self.data, pd.DataFrame)
        for key in self.date_cols:
            if self.data[key].isna().any():
                unix = self.data[key].astype("int64")
                unix[unix < 0] = np.nan  # NaT are -9e10
                unix = unix.interpolate(method="bfill").interpolate("pad")
                self.data[key] = pd.to_datetime(unix, unit="ns")

    def convert_float_int(self, data=None) -> pd.DataFrame:
        """
        Integer columns with NaN in them are interpreted as floats.
        In the beginning we check whether some columns should be integers,
        but we rely on different algorithms to take care of the NaN.
        Therefore, we only convert floats to integers at the very end
        """
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            self.data = data
        assert isinstance(self.data, pd.DataFrame)

        for key in self.bool_cols:
            if key in self.data:
                self.data[key] = pd.to_numeric(
                    self.data[key], errors="coerce", downcast="integer"
                )
        return self.data

    def _code_target_column(self, fit):
        """En- or decodes target column of `self.data`

        Parameters
        ----------
        fit : bool
            Whether to fit the encoder
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame but got '{type(self.data)}'.")
        if not self.target or self.target not in self.data:
            return

        # Get labels and encode / decode
        labels = self.data[self.target]  # type: ignore

        # Encode
        self.data[self.target] = self.encode_labels(
            labels, fit=fit, warn_unencodable=False
        )

    def encode_labels(self, labels, *, fit=True, warn_unencodable=True):
        """Encode labels to numerical dtype

        Parameters
        ----------
        labels : np.ndarray or pd.Series
            Labels to encode
        fit : bool
            Whether to (re)fit the label encoder
        warn_unencodable : bool
            Whether to warn when labels are assumed being for regression task

        Returns
        -------
        np.ndarray
            Encoded labels or original labels if unencodable

        Raises
        ------
        NotFittedError
            When no label encoder has yet been trained, i.e. `self._label_encodings` is
            empty
        """
        # Convert to pd.Series for convenience
        labels = pd.Series(labels)

        # It's probably a classification task
        if labels.dtype == object or labels.nunique() <= labels.size / 2:
            # Create encoding
            encoder = LabelEncoder()
            if fit is True:
                # Fit
                encoder.fit(labels)
                self._label_encodings = pd.Series(encoder.classes_).to_list()
            elif not self._label_encodings:
                raise NotFittedError("Encoder it not yet fitted")
            else:
                encoder.fit(self._label_encodings)
            # Encode
            return encoder.transform(labels)

        # It's probably a regression task, thus no encoding needed
        if warn_unencodable:
            warnings.warn(
                UserWarning(
                    "Labels are probably for regression. No encoding happened..."
                )
            )
        return labels.to_numpy()

    def decode_labels(self, labels, *, except_not_fitted=True):
        """Decode labels from numerical dtype to original value

        Parameters
        ----------
        labels : np.ndarray or pd.Series
            Labels to decode
        except_not_fitted : bool
            Whether to raise an exception when label encoder is not fitted

        Returns
        -------
        np.ndarray
            Decoded labels or original labels if label encoder is not fitted and
            `except_not_fitted` is True

        Raises
        ------
        NotFittedError
            When `except_not_fitted` is True and label encoder is not fitted
        """
        try:
            if len(self._label_encodings) == 0:
                raise NotFittedError(
                    "Encoder it not yet fitted. Try first calling `encode_target` "
                    "to set an encoding"
                )
            encoder = LabelEncoder()
            encoder.classes_ = np.array(self._label_encodings)
            return encoder.inverse_transform(labels)
        except NotFittedError as err:
            if except_not_fitted:
                raise err
            else:
                return labels.to_numpy() if isinstance(labels, pd.Series) else labels

    def _impute_columns(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        *** For production ***
        If a dataset is missing certain columns, this function looks at all registered
        columns and fills them with
        zeros.
        """
        if data is not None:
            assert isinstance(data, pd.DataFrame)
            self.data = data
        assert isinstance(self.data, pd.DataFrame)

        imputed = []
        for keys in [self.num_cols, self.cat_cols]:  # ignoring `self.date_cols`
            for key in [k for k in keys if k not in self.data]:
                self.data[key] = np.zeros(len(self.data))
                imputed.append(key)
        if len(imputed) > 0:
            warnings.warn(f"Imputed {len(imputed)} missing columns! {imputed}")
        return self.data

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
        self.date_cols = [f for f in self.date_cols if f in hash_features]
        self.num_cols = [f for f in self.num_cols if f in hash_features]
        self.bool_cols = [f for f in self.bool_cols if f in hash_features]
        self.cat_cols = [f for f in self.cat_cols if f in hash_features]
