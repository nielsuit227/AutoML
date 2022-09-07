#  Copyright (c) 2022 by Amplo.

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from amplo.utils import clean_feature_name
from amplo.utils.data import clean_keys
from amplo.utils.logging import logger

__all__ = ["DataProcessor"]


class DataProcessor:
    def __init__(
        self,
        target: str = None,
        float_cols: list = None,
        int_cols: list = None,
        date_cols: list = None,
        cat_cols: list = None,
        include_output: bool = True,
        missing_values: str = "interpolate",
        outlier_removal: str = "clip",
        z_score_threshold: int = 4,
        remove_constants: bool = True,
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
        float_cols : list
            Float columns
        int_cols : list
            Integer columns
        date_cols : list
            Date columns, all parsed to pd.datetime format
        cat_cols : list
            Categorical Columns. Currently, all one-hot encoded.
        include_output : bool
            Whether to include output in the data
        missing_values : str
            How to deal with missing values ("remove", "interpolate"or "mean")
        outlier_removal : str
            How to deal with outliers ("clip", "quantiles", "z-score"or "none")
        z_score_threshold : int
            If outlierRemoval="z-score", the threshold is adaptable
        remove_constants : bool
            If False, does not remove constant columns
        version : int
            Versioning the output files
        verbosity : int
            How much to print
        """
        # Tests
        mis_values_algo = ["remove_rows", "remove_cols", "interpolate", "mean", "zero"]
        if missing_values not in mis_values_algo:
            raise ValueError(
                "Missing values algorithm not implemented, pick from"
                f" {mis_values_algo}"
            )
        out_rem_algo = ["quantiles", "z-score", "clip", "none"]
        if outlier_removal not in out_rem_algo:
            raise ValueError(
                "Outlier Removal algorithm not implemented, pick from"
                f" {out_rem_algo}"
            )

        # Arguments
        self.version = version
        self.includeOutput = include_output
        self.target = (
            target if target is None else re.sub("[^a-z0-9]", "_", target.lower())
        )
        self.float_cols = (
            []
            if float_cols is None
            else [re.sub("[^a-z0-9]", "_", fc.lower()) for fc in float_cols]
        )
        self.int_cols = (
            []
            if int_cols is None
            else [re.sub("[^a-z0-9]", "_", ic.lower()) for ic in int_cols]
        )
        self.num_cols = self.float_cols + self.int_cols
        self.cat_cols = (
            []
            if cat_cols is None
            else [re.sub("[^a-z0-9]", "_", cc.lower()) for cc in cat_cols]
        )
        self.date_cols = (
            []
            if date_cols is None
            else [re.sub("[^a-z0-9]", "_", dc.lower()) for dc in date_cols]
        )
        if self.target in self.num_cols:
            self.num_cols.remove(self.target)

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
        self.removedDuplicateRows = 0
        self.removedDuplicateColumns = 0
        self.removedOutliers = 0
        self.imputedMissingValues = 0
        self.removedConstantColumns = 0

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

        # Clean Keys
        self.data = clean_keys(data)

        # Impute columns
        self._impute_columns()

        # Remove target
        if (
            fit
            and not self.includeOutput
            and self.target is not None
            and self.target in self.data
        ):
            self.data = self.data.drop(self.target, axis=1)

        if fit:
            # Remove Duplicates
            self.remove_duplicates()

            # Infer data-types
            self.infer_data_types()

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
        if self.verbosity > 0:
            logger.info(
                f"Data Cleaning Started, ({len(data)} x {len(data.keys())}) samples"
            )

        self._fit_transform(data, fit=True, remove_constants=remove_constants)

        # Finish
        self.is_fitted = True
        if self.verbosity > 0:
            logger.info(
                f"Processing completed, ({len(self.data)} x {len(self.data.keys())})"
                " samples returned"
            )

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

        return self.data

    def get_settings(self) -> dict:
        """
        Get settings to recreate fitted object.
        """
        assert self.is_fitted, "Object not yet fitted."
        settings = {
            "num_cols": self.num_cols,
            "float_cols": self.float_cols,
            "int_cols": self.int_cols,
            "date_cols": self.date_cols,
            "cat_cols": self.cat_cols,
            "_label_encodings": self._label_encodings,
            "missing_values": self.missing_values,
            "outlier_removal": self.outlier_removal,
            "z_score_threshold": self.z_score_threshold,
            "_means": None if self._means is None else self._means.to_json(),
            "_stds": None if self._stds is None else self._stds.to_json(),
            "_q1": None if self._q1 is None else self._q1.to_json(),
            "_q3": None if self._q3 is None else self._q3.to_json(),
            "dummies": self.dummies,
            "fit": {
                "imputed_missing_values": self.imputedMissingValues,
                "removed_outliers": self.removedOutliers,
                "removed_constant_columns": self.removedConstantColumns,
                "removed_duplicate_rows": self.removedDuplicateRows,
                "removed_duplicate_columns": self.removedDuplicateColumns,
            },
        }
        return settings

    def load_settings(self, settings: dict) -> None:
        """
        Loads settings from dictionary and recreates a fitted object
        """
        self.num_cols = settings.get("num_cols", [])
        self.float_cols = settings.get("float_cols", [])
        self.int_cols = settings.get("int_cols", [])
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

    def infer_data_types(self, data=None):
        """
        In case no data types are provided, this function infers the most likely data
        types
        """
        if len(self.cat_cols) == len(self.num_cols) == len(self.date_cols) == 0:
            # First cleanup
            self.data = (
                self.data.infer_objects() if data is None else data.infer_objects()
            )

            # Iterate through keys
            for key in self.data.keys():
                # Skip target
                if key == self.target:
                    continue

                # Integer
                elif pd.api.types.is_integer_dtype(self.data[key]):
                    self.int_cols.append(key)

                    if self.verbosity > 1:
                        logger.info(f"Found integer dtype: {key}")
                    continue

                # Float
                elif pd.api.types.is_float_dtype(self.data[key]):
                    self.float_cols.append(key)

                    if self.verbosity > 1:
                        logger.info(f"Found float dtype: {key}")
                    continue

                # Datetime
                elif pd.api.types.is_datetime64_any_dtype(self.data[key]):
                    self.date_cols.append(key)

                    if self.verbosity > 1:
                        logger.info(f"Found datetime dtype: {key}")
                    continue

                # Booleans
                elif pd.api.types.is_bool_dtype(self.data[key]):
                    self.int_cols.append(key)

                    if self.verbosity > 1:
                        logger.info(f"Found boolean dtype: {key}")
                    continue

                # Strings / Objects
                elif pd.api.types.is_object_dtype(self.data[key]):

                    # Check numerical
                    numeric = pd.to_numeric(
                        self.data[key], errors="coerce", downcast="integer"
                    )
                    if numeric.isna().sum() < len(self.data) * 0.3:
                        # Float
                        if pd.api.types.is_float_dtype(numeric):
                            self.float_cols.append(key)

                            if self.verbosity > 1:
                                logger.info(f"Found float dtype: {key}")

                        # Integer
                        if pd.api.types.is_integer_dtype(numeric):
                            self.int_cols.append(key)

                            if self.verbosity > 1:
                                logger.info(f"Found integer dtype: {key}")

                        # Update data and continue
                        self.data[key] = numeric
                        continue

                    # Check date (random subsample as it's expensive)
                    date = pd.to_datetime(
                        self.data[key].astype("str"),
                        errors="coerce",
                        infer_datetime_format=True,
                    )
                    if date.isna().sum() < 0.3 * len(self.data):
                        self.date_cols.append(key)
                        self.data[key] = date

                        if self.verbosity > 1:
                            logger.info(f"Found datetime dtype: {key}")
                        continue

                    # Check categorical variable
                    if self.data[key].nunique() < max(10, len(self.data) // 4):
                        self.cat_cols.append(key)

                        if self.verbosity > 1:
                            logger.info(f"Found categorical dtype: {key}")
                        continue

                # Else not found
                warnings.warn(f"Couldn't identify feature: {key}")

            # Set num cols for reverse compatibility
            self.num_cols = self.int_cols + self.float_cols

            # Print
            if self.verbosity > 0:
                logger.info(
                    f"Found {len(self.int_cols)} integer, {len(self.float_cols)} float,"
                    f" {len(self.cat_cols)} "
                    f"categorical and {len(self.date_cols)} datetime columns"
                )

        return

    def convert_data_types(
        self, data: pd.DataFrame = None, fit_categorical: bool = True
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
            self.data = data

        # Datetime columns
        for key in self.date_cols:
            self.data.loc[:, key] = pd.to_datetime(
                self.data[key], errors="coerce", infer_datetime_format=True, utc=True
            )

        # Integer columns
        for key in self.int_cols:
            self.data.loc[:, key] = pd.to_numeric(
                self.data[key], errors="coerce", downcast="integer"
            )

        # Float columns
        for key in self.float_cols:
            self.data.loc[:, key] = pd.to_numeric(
                self.data[key], errors="coerce", downcast="float"
            )

        # Categorical columns
        if fit_categorical:
            self.data = self._fit_cat_cols()
        else:
            assert self.is_fitted, (
                ".convert_data_types() was called with fit_categorical=False, while "
                "categorical encoder is not yet fitted."
            )
            self.data = self._transform_cat_cols()

        # We need everything to become numeric, so all that is not mentioned will be
        # handled as numeric
        all_cols = (
            self.float_cols
            + self.int_cols
            + self.date_cols
            + self.cat_cols
            + [self.target]
        )
        for key in self.data.keys():
            if key not in all_cols:
                self.data.loc[:, key] = pd.to_numeric(self.data[key], errors="coerce")

        return self.data

    def _fit_cat_cols(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Encoding categorical variables always needs a scheme. This fits the scheme.
        """
        if data is not None:
            self.data = data

        for key in self.cat_cols:
            # Clean the categorical variables
            self.data[key] = self.data[key].apply(clean_feature_name)

            # Get dummies, convert & store
            dummies = pd.get_dummies(
                self.data[key], prefix=key, dummy_na=self.data[key].isna().sum() > 0
            )
            self.data = pd.concat([self.data.drop(key, axis=1), dummies], axis=1)
            self.dummies[key] = dummies.keys().tolist()
        return self.data

    def _transform_cat_cols(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Converts categorical variables according to fitted scheme.
        """
        if data is not None:
            self.data = data

        for key in self.cat_cols:
            # Clean column
            self.data[key] = self.data[key].apply(clean_feature_name)

            # Transform
            value = self.dummies[key]
            dummies = [i[len(key) + 1 :] for i in value]
            self.data[value] = (
                np.equal.outer(self.data[key].astype("str").values, dummies) * 1
            )
            self.data = self.data.drop(key, axis=1)
        return self.data

    def remove_duplicates(
        self, data: pd.DataFrame = None, rows: bool = True
    ) -> pd.DataFrame:
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
            self.data = data

        # Note down
        n_rows, n_columns = len(self.data), len(self.data.keys())

        # Remove Duplicates
        if rows:
            self.data = self.data.drop_duplicates()
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]

        # Note
        self.removedDuplicateColumns = n_columns - len(self.data.keys())
        self.removedDuplicateRows = n_rows - len(self.data)
        if self.verbosity > 0 or (
            self.removedDuplicateColumns != 0 or self.removedDuplicateRows != 0
        ):
            logger.info(
                f"Removed {self.removedDuplicateColumns} duplicate columns and"
                f" {self.removedDuplicateRows} duplicate rows"
            )

        return self.data

    def remove_constants(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Removes constant columns
        """
        if data is not None:
            self.data = data
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
        self.removedConstantColumns = columns - len(self.data.keys())
        if self.verbosity > 0 or self.removedConstantColumns != 0:
            logger.info(f"Removed {self.removedConstantColumns} constant columns.")

        return self.data

    def fit_outliers(self, data: pd.DataFrame = None):
        """
        Checks outliers
        """
        if data is not None:
            self.data = data

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
        self, data: pd.DataFrame = None, fit: bool = True
    ) -> pd.DataFrame:
        """
        Removes outliers
        """
        if data is not None:
            self.data = data

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
            self.removedOutliers = (
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
            self.removedOutliers = (
                (z_score > self.z_score_threshold).sum().sum().tolist()
            )
            self.data[self.num_cols] = self.data[self.num_cols].mask(
                z_score > self.z_score_threshold
            )

        # With clipping
        elif self.outlier_removal == "clip":
            self.removedOutliers = (
                (self.data[self.num_cols] > 1e12).sum().sum()
                + (self.data[self.num_cols] < -1e12).sum().sum()
            ).tolist()
            self.data[self.num_cols] = self.data[self.num_cols].clip(
                lower=-1e12, upper=1e12
            )
        return self.data

    def remove_missing_values(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fills missing values (infinities and "not a number"s)
        """
        if data is not None:
            self.data = data

        # Replace infinities
        self.data = self.data.replace([np.inf, -np.inf], np.nan)

        # Note
        self.imputedMissingValues = self.data[self.num_cols].isna().sum().sum().tolist()
        if self.verbosity > 0 or self.imputedMissingValues != 0:
            logger.info(f"Imputed {self.imputedMissingValues} missing values.")

        # Removes all rows with missing values
        if self.missing_values == "remove_rows":
            self.data = self.data[self.data.isna().sum(axis=1) == 0]

        # Removes all columns with missing values
        elif self.missing_values == "remove_cols":
            self.data = self.data.loc[:, self.data.isna().sum(axis=0) == 0]

        # Fills all missing values with zero
        elif self.missing_values == "zero":
            self.data = self.data.fillna(0)

        # Linearly interpolates missing values
        elif self.missing_values == "interpolate":

            # Columns which are present with >10% missing values are not interpolated
            zero_keys = self.data.keys()[
                self.data.isna().sum() / len(self.data) > 0.1
            ].tolist()

            # Get all non-date_cols & interpolate
            ik = np.setdiff1d(self.data.keys().to_list(), self.date_cols + zero_keys)
            self.data[ik] = self.data[ik].interpolate(limit_direction="both")

            # Fill date columns
            for key in self.date_cols:
                if self.data[key].isna().sum() != 0:
                    # Interpolate
                    ints = pd.Series(self.data[key].values.astype("int64"))
                    ints[ints < 0] = np.nan
                    self.data[key] = pd.to_datetime(ints.interpolate(), unit="ns")

            # Fill rest (date & more missing values cols)
            if self.data.isna().sum().sum() != 0:
                self.data = self.data.fillna(0)

        # Fill missing values with column mean
        elif self.missing_values == "mean":
            self.data = self.data.fillna(self.data.mean())

            # Need to be individual for some reason
            for key in self.date_cols:
                self.data[key] = self.data[key].fillna(self.data[key].mean())

        return self.data

    def convert_float_int(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Integer columns with NaN in them are interpreted as floats.
        In the beginning we check whether some columns should be integers,
        but we rely on different algorithms to take care of the NaN.
        Therefore, we only convert floats to integers at the very end
        """
        if data is not None:
            self.data = data

        for key in self.int_cols:
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

        if self.target not in self.data:
            return

        # Get labels and encode / decode
        labels = self.data.loc[:, self.target]

        # Encode
        self.data.loc[:, self.target] = self.encode_labels(
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

        # It"s probably a classification task
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

        # It"s probably a regression task, thus no encoding needed
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

    def _impute_columns(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        *** For production ***
        If a dataset is missing certain columns, this function looks at all registered
        columns and fills them with
        zeros.
        """
        if data is not None:
            self.data = data

        imputed = []
        for keys in [self.num_cols, self.date_cols, self.cat_cols]:
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
        self.int_cols = [f for f in self.int_cols if f in hash_features]
        self.float_cols = [f for f in self.float_cols if f in hash_features]
        self.cat_cols = [f for f in self.cat_cols if f in hash_features]
