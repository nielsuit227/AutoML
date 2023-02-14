#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Any

import dateutil
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from amplo.base import BaseTransformer, LoggingMixin
from amplo.utils.data import pandas_to_polars, polars_to_pandas
from amplo.utils.util import check_dtypes, clean_feature_name

__all__ = ["clean_feature_name", "DataProcessor"]


def _to_datetime(value):
    if isinstance(value, datetime.datetime):
        return value
    elif isinstance(value, str):
        try:
            return dateutil.parser.parse(value)  # type: ignore
        except Exception:
            return None
    else:
        return None


def pl_masked_fill(
    df: pl.DataFrame, mask: pl.DataFrame, replace_value=None
) -> pl.DataFrame:
    """
    Masked fill a DataFrame with 'replace_value'.

    This function only returns columns that were present in the 'mask'.

    Parameters
    ----------
    df : pl.DataFrame
        Data to be manipulated.
    mask : pl.DataFrame
        Mask with same or subset of 'df' columns.
    replace_value : Any, optional
        Fill value, by default None

    Returns
    -------
    pl.DataFrame
        Mask-filled DataFrame.
    """
    out = pl.concat(
        [df, mask.select(pl.all().suffix("_mask"))],
        how="horizontal",
    ).select(
        [
            pl.when(pl.col(f"{c}_mask"))
            .then(replace_value)
            .otherwise(pl.col(c))
            .alias(c)
            for c in mask.columns
        ]
    )
    return out


class DataProcessor(BaseTransformer, LoggingMixin):
    """
    Preprocessor. Cleans a dataset into a workable format.

    Deals with outliers, missing values, duplicate rows,
    data types (numerical, categorical and dates), NaN, and infinities.

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
    verbose : int
        How much to print
    """

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
        verbose: int = 1,
    ):
        BaseTransformer.__init__(self)
        LoggingMixin.__init__(self, verbose=verbose)

        # Type checks
        check_dtypes(
            ("target", target, (type(None), str)),
            ("include_output", include_output, bool),
            ("drop_datetime", drop_datetime, bool),
            ("drop_constants", drop_constants, bool),
            ("drop_duplicate_rows", drop_duplicate_rows, bool),
            ("z_score_threshold", z_score_threshold, int),
            ("drop_duplicate_rows", drop_duplicate_rows, bool),
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
        self.num_cols_: list[str] = []
        self.bool_cols_: list[str] = []
        self.cat_cols_: list[str] = []
        self.date_cols_: list[str] = []
        self.dummies_: dict[str, list[str]] = {}
        self.q1_: pd.Series | None = None
        self.q3_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None
        self.label_encodings_: list[str] = []
        self.rename_dict_: dict[str, str]

        # Info for Documenting
        self.is_fitted_ = False
        self.removed_duplicate_rows_ = 0
        self.removed_duplicate_columns_ = 0
        self.removed_outliers_ = 0
        self.imputed_missing_values_ = 0
        self.removed_constant_columns_ = 0

    def fit(self, data: pd.DataFrame):
        """
        Fits the data processor.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the transformer.
        """
        self.transform(data, fit=True)
        return self

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the data processor.

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        return self.transform(data, fit=True)

    def transform(self, data: pd.DataFrame, *, fit: bool = False) -> pd.DataFrame:
        """
        (Fits and) transforms the data processor.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        fit : bool, optional
            Whether to fit the data, by default False

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        if fit:
            self.logger.info("Data processor starts fitting and transforming...")
        elif not self.is_fitted_:
            raise NotFittedError
        else:
            self.logger.info("Data processor starts transforming...")

        # Convert to polars
        self.logger.debug("Convert pandas data to polars.")
        pl_data, index_names = pandas_to_polars(data, index_prefix="idx_lvl")
        work_index_names = list(index_names)

        # Data processing
        pl_data = self.clean_column_names(pl_data, work_index_names, fit=fit)

        if fit:
            pl_data = self.remove_duplicates(
                pl_data, work_index_names, rows=self.drop_duplicate_rows
            )
            pl_data = self.infer_data_types(pl_data, work_index_names)
        else:
            pl_data = self._impute_columns(pl_data)

        pl_data = self.convert_data_types(
            pl_data, work_index_names, fit_categorical=fit
        )
        pl_data = self.remove_outliers(pl_data, fit=fit)
        pl_data = self.fill_missing_values(pl_data)

        if fit and self.drop_constants:
            pl_data = self.remove_constants(pl_data)

        pl_data = self.encode_labels(pl_data, fit=fit)

        # Convert back to pandas and restore index
        self.logger.debug("Convert polars data back do pandas.")
        data = polars_to_pandas(pl_data, index_names)

        # Finish
        if fit:
            self.is_fitted_ = True
            self.logger.info("Data processor finished fitting and transforming.")
        else:
            self.logger.info("Data processor finished transforming.")

        return data

    def clean_column_names(
        self, data: pl.DataFrame, index_cols: list[str], fit: bool = False
    ) -> pl.DataFrame:
        """Ensures there are no strange characters in feature names."""
        self.logger.debug("Cleaning column names.")

        if fit:
            return self._fit_clean_column_names(data, index_cols)
        else:
            return self._transform_clean_column_names(data, index_cols)

    def _fit_clean_column_names(
        self, data: pl.DataFrame, index_cols: list[str]
    ) -> pl.DataFrame:
        self.logger.debug("Start fitting and cleaning column names.")

        # Transform
        self.rename_dict_ = {
            c: clean_feature_name(c)
            for c in map(str, data.columns)
            if c not in index_cols  # we don't want to fit the index names
        }
        data = self._transform_clean_column_names(data, index_cols)

        # Could be that this is introducing duplicate columns, e.g.
        # when the data is partially cleaned.

        # Update target
        if self.target is not None:
            self.target = clean_feature_name(self.target)

        # Remove target from data if need be
        if not self.include_output and self.target is not None and self.target in data:
            data = data.drop(self.target)

        self.logger.debug("Finished fitting and cleaning column names.")
        return data

    def _transform_clean_column_names(
        self, data: pl.DataFrame, index_cols: list[str]
    ) -> pl.DataFrame:
        self.logger.debug("Start cleaning column names.")

        # Polars does not allow having multiple columns with the same name.
        # Here we invert the rename dictionary to see which introduce duplicates.
        inv_rename_dict = defaultdict(list)
        for old_name, new_name in self.rename_dict_.items():
            inv_rename_dict[new_name].append(old_name)

        # Check out which renamings introduce duplicates (not allowed by polars!)
        safe_rename_dict: dict[str, str] = {}
        duplicated_clean: set[str] = set()
        for old_name, new_name in self.rename_dict_.items():
            if len(inv_rename_dict[new_name]) > 1:
                duplicated_clean = duplicated_clean.union({new_name})
            else:
                safe_rename_dict[old_name] = new_name

        # Warn for duplicate renaming
        if duplicated_clean:
            warn_msg = (
                f"{len(duplicated_clean)} columns need inspection. Cleaning the column "
                f"names would introduce duplicated column names which is not supported "
                f"by polars. List of critical columns: {sorted(duplicated_clean)}"
            )
            self.logger.warning(warn_msg)

        # Handle safe renaming
        data = data.rename(safe_rename_dict)

        # Handle duplicate renaming
        for new_name in duplicated_clean:
            old_names = inv_rename_dict[new_name]

            # Use first column as base and fill nulls with all other columns
            expr = pl.col(old_names[0])
            for old_name in old_names[1:]:
                expr = expr.fill_null(pl.col(old_name))

            # Apply
            data = data.with_column(expr.alias(new_name))

        # Clean up: which cols one should keep
        keep_cols = {*safe_rename_dict.values(), *duplicated_clean}
        if self.target not in data:
            keep_cols -= {self.target}
        data = data[[*index_cols, *keep_cols]]

        self.logger.debug("Finished cleaning column names.")
        return data

    def remove_duplicates(
        self, data: pl.DataFrame, index_names: list[str], rows: bool = False
    ) -> pl.DataFrame:
        """
        Removes duplicate columns and rows.

        Parameters
        ----------
        data : pl.DataFrame
            Input data
        index_names : list[str]
            Column names of the index columns.
        rows : bool
            Whether to remove duplicate rows. This is only recommended with data that
            has no temporal structure, and only for training data.
        """
        self.logger.debug("Start removing duplicate columns (and rows).")

        # Note down
        n_rows, n_columns = data.shape

        # Remove duplicate rows
        if rows:
            subset = [c for c in data.columns if c not in index_names]
            data = data.unique(subset=subset)  # equivalent of 'pandas.drop_duplicates'

        # Note
        self.removed_duplicate_rows = rdr = n_rows - data.shape[0]
        self.removed_duplicate_columns_ = rdc = n_columns - data.shape[1]
        self.logger.debug(f"Finished removing {rdc} duplicate columns and {rdr} rows.")

        return data

    def infer_data_types(
        self, data: pl.DataFrame, index_names: list[str]
    ) -> pl.DataFrame:
        """
        In case no data types are provided, this function infers the most likely data
        types

        parameters
        ----------
        data : pl.DataFrame
            Data to infer data types.
        index_names : list[str]
            Column names of the index columns in order not to drop them.

        returns
        -------
        data : pl.DataFrame
        """
        self.logger.debug("Inferring data types.")

        # Initialize
        self.num_cols_ = []
        self.bool_cols_ = []
        self.date_cols_ = []
        self.cat_cols_ = []

        # Iterate through keys
        for key in data.columns:

            # Skip target and index -- we don't want to convert it!
            if key == self.target:
                self.logger.debug(f"- Skipped target column '{key}'.")
                continue
            elif key in index_names:
                self.logger.debug(f"- Skipped index column '{key}'.")
                continue

            # Extract column
            f = data[key]
            f_type = None

            # Integer and Float
            if f.dtype in (
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
            ):
                f_type = "numerical"

            # Datetime
            elif f.dtype in (pl.Date, pl.Datetime):
                f_type = "datetime"

            # Booleans
            elif f.dtype == pl.Boolean:
                f_type = "boolean"

            # Strings / Objects
            elif f.dtype in (pl.Object, pl.Utf8):

                # Check numerical
                numeric = f.cast(pl.Float32, strict=False)
                if numeric.null_count() < 0.3 * len(numeric):
                    f_type = "numerical"
                    # Update data
                    data = data.with_column(numeric.alias(key))

                # Check categorical
                elif f.n_unique() < min(max(10, len(data) // 4), 50):
                    f_type = "categorical"

                # Check datetime/time
                elif not self.drop_datetime:
                    datetime = f.apply(_to_datetime)
                    if datetime.null_count() < 0.3 * len(datetime):
                        f_type = "datetime"
                        # Update data
                        data = data.with_column(datetime.alias(key))

            # Report type of feature
            if f_type == "numerical":
                self.num_cols_.append(key)
            elif f_type == "boolean":
                self.bool_cols_.append(key)
            elif f_type == "categorical":
                self.cat_cols_.append(key)
            elif f_type == "datetime":
                self.date_cols_.append(key)
            else:
                self.logger.warning(f"Couldn't identify type of column '{key}'.")
                continue

            self.logger.debug(f"- Found {f_type} column '{key}'.")

        self.logger.debug("Finished inferring data types.")
        return data

    def convert_data_types(
        self, data: pl.DataFrame, index_names: list[str], fit_categorical: bool = True
    ) -> pl.DataFrame:
        """
        Cleans up the data types of all columns.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        index_names : list[str]
            Column names of the index columns in order not to drop them.
        fit_categorical : bool
            Whether to fit the categorical encoder

        Returns
        -------
        pd.DataFrame
            Cleaned input data
        """
        self.logger.debug("Converting data types")

        # Drop unused columns & datetime columns
        keep_cols = [
            *index_names,
            *self.num_cols_,
            *self.date_cols_,
            *self.bool_cols_,
            *self.cat_cols_,
            self.target,
        ]
        data = data.drop([k for k in data.columns if k not in keep_cols])

        if self.date_cols_ and self.drop_datetime:
            self.logger.warning(
                f"Data contains datetime columns but are removed: '{self.date_cols_}'",
            )
            data = data.drop(self.date_cols_)
        elif self.date_cols_ and not self.drop_datetime:
            # Convert to datetime format
            data = data.with_columns(
                [pl.col(c).apply(_to_datetime) for c in self.date_cols_]
            )

        # Integer columns
        if self.bool_cols_:
            data = data.with_columns(
                [pl.col(col).cast(pl.UInt8, strict=False) for col in self.bool_cols_]
            )

        # Float columns
        if self.num_cols_:
            data = data.with_columns(
                [pl.col(col).cast(pl.Float32, strict=False) for col in self.num_cols_]
            )

        # Categorical columns
        self.logger.debug("(Fit-)Transforming categorical columns")
        if fit_categorical:
            data = self._fit_transform_cat_cols(data)
        elif self.is_fitted_:
            data = self._transform_cat_cols(data)
        else:
            raise NotFittedError(
                ".convert_data_types() was called with fit_categorical=False, while "
                "categorical encoder is not yet fitted."
            )

        self.logger.debug("Finished converting data types.")
        return data

    def _fit_transform_cat_cols(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Encoding categorical variables always needs a scheme. This fits the scheme.

        Parameters
        ----------
        data : pl.DataFrame
            Input data

        Returns
        -------
        pl.DataFrame
            Cleaned input data
        """
        # Clean values of cat_cols
        # Note that 'null' and 'nan' are ignored and thus not stringified
        cat_data = data[self.cat_cols_].select([pl.all().apply(clean_feature_name)])
        data = data.with_columns([*cat_data])

        # One-hot encode each categorical column, including 'null's
        self.dummies_ = {}
        for col in self.cat_cols_:
            # Get one-hot encoding (dummies). Note that we have already cleaned it.
            series = data.drop_in_place(col)  # 'pop' column
            one_hot = series.to_dummies()

            # Update data
            if set(one_hot.columns).intersection(data.columns):
                raise RuntimeError(
                    f"One-hot encoding of the column '{col}' introduces duplicated "
                    f"column names: {set(one_hot.columns).intersection(data.columns)}"
                )
            data = data.with_columns([*one_hot])

            # Store reference
            self.dummies_[col] = one_hot.columns

        self.logger.debug("Finished fit-transforming categorical columns")
        return data

    def _transform_cat_cols(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Converts categorical variables according to fitted scheme.

        Parameters
        ----------
        data : pl.DataFrame
            Input data

        Returns
        -------
        pl.DataFrame
            Cleaned input data
        """
        # Clean values of cat_cols
        # Note that 'null' and 'nan' are ignored and thus not stringified
        cat_data = data[self.cat_cols_].select([pl.all().apply(clean_feature_name)])
        data = data.with_columns([*cat_data])

        # One-hot encode each categorical column. Note that we have already cleaned it.
        data = data.to_dummies(columns=self.cat_cols_)

        # Insert missing categories
        missing_categories = [
            cat
            for col in self.cat_cols_
            for cat in self.dummies_[col]
            if cat not in data.columns
        ]
        if missing_categories:
            data = data.with_columns(
                # insert new, zero filled column
                [pl.lit(0).cast(pl.UInt8).alias(cat) for cat in missing_categories]
            )

        self.logger.debug("Transformed categorical columns.")
        return data

    def remove_constants(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Removes constant columns

        Parameters
        ----------
        data : pl.DataFrame
            Input data

        Returns
        -------
        pl.DataFrame
            Cleaned input data
        """
        self.logger.debug("Removing constants.")

        # Find constant columns and make a note
        constant_columns = [c for c in data.columns if data[c].unique().shape[0] == 1]
        self.removed_constant_columns_ = len(constant_columns)

        # Remove those columns
        self.logger.debug(f"Removed {len(constant_columns)} constant columns.")
        return data.drop(constant_columns)

    def remove_outliers(self, data: pl.DataFrame, fit: bool = False) -> pl.DataFrame:
        """
        (Fits and) removes outliers according to `self.outlier_removal`.

        Parameters
        ----------
        data : pl.DataFrame
            Input data
        fit : bool
            Whether to fit the data, by default False

        Returns
        -------
        pl.DataFrame
            Cleaned input data
        """
        self.logger.debug("Removing outliers.")

        num_data = data[self.num_cols_]
        if num_data.is_empty():
            return data

        # With quantiles
        if self.outlier_removal == "quantiles":
            if fit:
                q1 = num_data.quantile(0.05)
                q3 = num_data.quantile(0.95)
                self.q1_ = q1.to_dicts()
                self.q3_ = q3.to_dicts()
            else:
                q1 = pl.DataFrame(self.q1_)
                q3 = pl.DataFrame(self.q3_)

            # Mark as True where values are below q1 or above q3
            mask = num_data.select(
                [(pl.col(c) < q1[c]) | (pl.col(c) > q3[c]) for c in self.num_cols_]
            )
            # Count how many values have been replaced by 'null'
            self.removed_outliers_ = int(mask.sum().to_numpy().sum())
            # Apply
            num_data = pl_masked_fill(num_data, mask, replace_value=None)
            data = data.with_columns([*num_data])

        # With z-score
        elif self.outlier_removal == "z-score":
            if fit:
                means = num_data.mean()
                stds = num_data.std()
                stds = stds.with_columns(
                    [
                        pl.when(pl.col(c) < 0.1).then(0.1).otherwise(pl.col(c)).alias(c)
                        for c in stds.columns
                    ]
                )
                self.means_ = means.to_dicts()
                self.stds_ = stds.to_dicts()
            else:
                means = pl.DataFrame(self.means_)
                stds = pl.DataFrame(self.stds_)

            # Mark as True where z_score is above threshold
            mask = num_data.select(
                [
                    (pl.col(c) - means[c]) / stds[c] > self.z_score_threshold
                    for c in num_data.columns
                ]
            )
            # Count how many values have been replaced by 'null'
            self.removed_outliers_ = int(mask.sum().to_numpy().sum())
            # Apply
            num_data = pl_masked_fill(num_data, mask, replace_value=None)
            data = data.with_columns([*num_data])

        # With clipping
        elif self.outlier_removal == "clip":
            thr = 1e12
            # Count how many values have been clipped from below/above
            self.removed_outliers_ = int(
                pl.concat([num_data < -thr, num_data > thr]).sum().to_numpy().sum()
            )
            # Clip values and apply
            data = data.with_columns(
                [pl.col(c).clip(-thr, thr) for c in self.num_cols_]
            )

        self.logger.debug("Finished removing outliers.")
        return data

    def fill_missing_values(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Fills missing values (infinities and "not a number"s)

        Parameters
        ----------
        data : pl.DataFrame
            Input data

        Returns
        -------
        pl.DataFrame
            Cleaned input data
        """
        self.logger.debug("Start filling missing values.")

        # Replace infinities
        inf_mask = data.select(([pl.col(c).abs() == np.inf for c in self.num_cols_]))
        data = data.with_columns([*pl_masked_fill(data, inf_mask, None)])
        # Replace nan
        data = data.fill_nan(None)

        # Note
        self.imputed_missing_values_ = int(data.null_count().to_numpy().sum())

        # Removes all rows with missing values
        if self.missing_values == "remove_rows":
            data = data.drop_nulls()

        # Removes all columns with missing values
        elif self.missing_values == "remove_cols":
            null_count = data.null_count().to_pandas().iloc[0]
            null_columns = null_count[null_count > 0].index.to_list()
            data = data.drop(null_columns)

        # Fills all missing values with zero
        elif self.missing_values == "zero":
            data = data.fill_null(0)

        # Mean and Interpolate require more than 1 value, use zero if less
        elif self.missing_values in ("interpolate", "mean") and len(data) <= 1:
            data = data.fill_null(0)

        # Linearly interpolates missing values
        elif self.missing_values == "interpolate":
            # Interpolate
            data = data.interpolate()  # also interpolates dates
            # Forward and backward fill
            data = data.fill_null(strategy="forward").fill_null(strategy="backward")

        # Fill missing values with column mean
        elif self.missing_values == "mean":
            data = data.fill_null(strategy="mean")

        # 'fill_null(0)' has no effect on datetime data
        if data.height > 1:
            data = data.fill_null(strategy="mean")
        else:
            # Simply drop columns that only have NaNs
            data = data[[s.name for s in data if s.null_count() != data.height]]

        self.logger.debug(f"Removed {self.imputed_missing_values_} missing values.")
        return data

    def encode_labels(self, data: pl.DataFrame, fit: bool) -> pl.DataFrame:
        """
        En- or decodes target column of `data`

        Parameters
        ----------
        data : pl.DataFrame
            input data
        fit : bool
            Whether to (re)fit the label encoder

        Returns
        -------
        data : pl.DataFrame
            With the encoded labels
        """
        self.logger.debug("Encoding labels")

        # Get labels and encode / decode
        if not self.target or self.target not in data:
            return data
        if not self.include_output:
            return data.drop(self.target)

        # Split output
        labels = data[self.target]

        # Check whether it's classification
        if labels.dtype == pl.Utf8 or labels.n_unique() <= min(labels.len() / 2, 50):
            # Create encoder
            encoder = LabelEncoder()
            if fit is True:
                encoder.fit(labels)
                self.label_encodings_ = list(encoder.classes_)
            elif not self.label_encodings_:
                raise NotFittedError
            else:
                encoder.fit(self.label_encodings_)

            # Encode
            enc_labels = pl.Series(encoder.transform(labels)).alias(self.target)
            return data.with_column(enc_labels)

        # It's probably a regression task, thus no encoding needed
        else:
            self.logger.warning(
                "Labels are probably for regression. No encoding happened..."
            )
            return data

    def decode_labels(self, data: npt.NDArray[Any]) -> pd.Series:
        """
        Decode labels from numerical dtype to original value

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
        self.logger.debug("Decoding labels.")

        # Checks
        if not self.label_encodings_:
            raise NotFittedError

        # Create encoder
        encoder = LabelEncoder().fit(self.label_encodings_)

        # Decode
        return pd.Series(encoder.inverse_transform(data))

    def _impute_columns(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        *** For production ***
        If a dataset is missing certain columns, this function looks at all registered
        columns and fills them with
        zeros.

        Parameters
        ----------
        data : pl.DataFrame
            Input data

        Returns
        -------
        data : pl.DataFrame
        """
        self.logger.debug("Imputing columns")

        # Impute
        required_cols = [
            *self.date_cols_,
            *self.num_cols_,
            *self.bool_cols_,
            *self.cat_cols_,
        ]
        missing_cols = [c for c in required_cols if c not in data]
        data = data.with_columns([pl.lit(0).alias(c) for c in missing_cols])

        # Warn
        if len(missing_cols) > 0:
            self.logger.warning(
                f"Imputed {len(missing_cols)} missing columns! >> {missing_cols}"
            )
        return data

    def prune_features(self, features: list[str]) -> None:
        """
        For use with AutoML.Pipeline. We practically never use all features. Yet this
        processor imputes any missing features. This causes redundant operations,
        memory, and warnings. This function prunes the features to avoid that.

        parameters
        ----------
        features : list
            Required features (NOTE: include required features for extracted)
        """
        self.logger.debug("Pruning dataprocessor features.")

        if not self.is_fitted_:
            raise NotFittedError()

        hash_features = {k: 0 for k in features}
        self.date_cols_ = [f for f in self.date_cols_ if f in hash_features]
        self.num_cols_ = [f for f in self.num_cols_ if f in hash_features]
        self.bool_cols_ = [f for f in self.bool_cols_ if f in hash_features]
        self.cat_cols_ = [f for f in self.cat_cols_ if f in hash_features]
        self.rename_dict_ = {
            k: v for k, v in self.rename_dict_.items() if v in features
        }
