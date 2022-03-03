import re
import warnings
import numpy as np
import pandas as pd
from Amplo.Utils import clean_keys


class DataProcesser:

    def __init__(self,
                 target: str = None,
                 float_cols: list = None,
                 int_cols: list = None,
                 date_cols: list = None,
                 cat_cols: list = None,
                 include_output: bool = True,
                 missing_values: str = 'interpolate',
                 outlier_removal: str = 'clip',
                 z_score_threshold: int = 4,
                 remove_constants: bool = True,
                 version: int = 1,
                 verbosity: int = 0,
                 ):
        """
        Preprocessing Class. Cleans a dataset into a workable format.
        Deals with Outliers, Missing Values, duplicate rows, data types (floats, categorical and
        dates), Not a Numbers, Infinities.

        Parameters
        ----------
        target str: Column name of target variable
        float_cols list: Float columns
        int_cols list: Integer columns
        date_cols list: Date columns, all parsed to pd.datetime format
        cat_cols list: Categorical Columns. Currently all one-hot encoded.
        include_output bool: Whether to include output in the data
        missing_values str: How to deal with missing values ('remove', 'interpolate' or 'mean')
        outlier_removal str: How to deal with outliers ('clip', 'quantiles', 'z-score' or 'none')
        z_score_threshold int: If outlierRemoval='z-score', the threshold is adaptable, default=4.
        remove_constants bool: If false, does not remove constant columns
        version int: Versioning the output files
        verbosity int: How much to print
        """
        # Tests
        mis_values_algo = ['remove_rows', 'remove_cols', 'interpolate', 'mean', 'zero']
        assert missing_values in mis_values_algo, \
            'Missing values algorithm not implemented, pick from {}'.format(', '.join(mis_values_algo))
        out_rem_algo = ['quantiles', 'z-score', 'clip', 'none']
        assert outlier_removal in out_rem_algo, \
            'Outlier Removal algorithm not implemented, pick from {}'.format(', '.join(out_rem_algo))

        # Arguments
        self.version = version
        self.includeOutput = include_output
        self.target = target if target is None else re.sub("[^a-z0-9]", '_', target.lower())
        self.float_cols = [] if float_cols is None else [re.sub('[^a-z0-9]', '_', fc.lower()) for fc in float_cols]
        self.int_cols = [] if int_cols is None else [re.sub('[^a-z0-9]', '_', ic.lower()) for ic in int_cols]
        self.num_cols = self.float_cols + self.int_cols
        self.cat_cols = [] if cat_cols is None else [re.sub('[^a-z0-9]', '_', cc.lower()) for cc in cat_cols]
        self.date_cols = [] if date_cols is None else [re.sub('[^a-z0-9]', '_', dc.lower()) for dc in date_cols]
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

        # Info for Documenting
        self.is_fitted = False
        self.verbosity = verbosity
        self.removedDuplicateRows = 0
        self.removedDuplicateColumns = 0
        self.removedOutliers = 0
        self.imputedMissingValues = 0
        self.removedConstantColumns = 0

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits this data cleaning module and returns the transformed data.


        Parameters
        ----------
        data [pd.DataFrame]: Input data

        Returns
        -------
        data [pd.DataFrame]: Cleaned input data
        """
        if self.verbosity > 0:
            print('[AutoML] Data Cleaning Started, ({} x {}) samples'.format(len(data), len(data.keys())))

        # Clean Keys
        self.data = clean_keys(data)

        # Remove target
        if not self.includeOutput and self.target is not None and self.target in data:
            data = data.drop(self.target, axis=1)

        # Remove Duplicates
        self.remove_duplicates()

        # Infer data-types
        self.infer_data_types()

        # Convert data types
        self.convert_data_types(fit_categorical=True)

        # Remove outliers
        self.remove_outliers(fit=True)

        # Remove missing values
        self.remove_missing_values()

        # Remove Constants
        self.remove_constants()

        # Convert integer columns
        self.convert_float_int()

        # Clean target
        self.clean_target()

        # Finish
        self.is_fitted = True
        if self.verbosity > 0:
            print('[AutoML] Processing completed, ({} x {}) samples returned'.format(len(data), len(data.keys())))
        return self.data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Function that takes existing settings (including dummies), and transforms new data.

        Parameters
        ----------
        data [pd.DataFrame]: Input data

        Returns
        -------
        data [pd.DataFrame]: Cleaned input data
        """
        assert self.is_fitted, "Transform only available for fitted objects, run .fit_transform() first."

        # Clean Keys
        self.data = clean_keys(data)

        # Impute columns
        self._impute_columns()

        # Remove duplicates
        self.remove_duplicates(rows=False)

        # Convert data types
        self.convert_data_types(fit_categorical=False)

        # Remove outliers
        self.remove_outliers(fit=False)

        # Remove missing values
        self.remove_missing_values()

        # Convert integer columns
        self.convert_float_int()

        return self.data

    def get_settings(self) -> dict:
        """
        Get settings to recreate fitted object.
        """
        assert self.is_fitted, "Object not yet fitted."
        return {
            'num_cols': self.num_cols,
            'float_cols': self.float_cols,
            'int_cols': self.int_cols,
            'date_cols': self.date_cols,
            'cat_cols': self.cat_cols,
            'missing_values': self.missing_values,
            'outlier_removal': self.outlier_removal,
            'z_score_threshold': self.z_score_threshold,
            '_means': None if self._means is None else self._means.to_json(),
            '_stds': None if self._stds is None else self._stds.to_json(),
            '_q1': None if self._q1 is None else self._q1.to_json(),
            '_q3': None if self._q3 is None else self._q3.to_json(),
            'dummies': self.dummies,
            'fit': {
                'imputed_missing_values': self.imputedMissingValues,
                'removed_outliers': self.removedOutliers,
                'removed_constant_columns': self.removedConstantColumns,
                'removed_duplicate_rows': self.removedDuplicateRows,
                'removed_duplicate_columns': self.removedDuplicateColumns,
            }
        }

    def load_settings(self, settings: dict) -> None:
        """
        Loads settings from dictionary and recreates a fitted object
        """
        self.num_cols = settings['num_cols'] if 'num_cols' in settings else []
        self.float_cols = settings['float_cols'] if 'float_cols' in settings else []
        self.int_cols = settings['int_cols'] if 'int_cols' in settings else []
        self.cat_cols = settings['cat_cols'] if 'cat_cols' in settings else []
        self.date_cols = settings['date_cols'] if 'date_cols' in settings else []
        self.missing_values = settings['missing_values'] if 'missing_values' in settings else []
        self.outlier_removal = settings['outlier_removal'] if 'outlier_removal' in settings else []
        self.z_score_threshold = settings['z_score_threshold'] if 'z_score_threshold' in settings else []
        self._means = None if settings['_means'] is None else pd.read_json(settings['_means'])
        self._stds = None if settings['_stds'] is None else pd.read_json(settings['_stds'])
        self._q1 = None if settings['_q1'] is None else pd.read_json(settings['_q1'])
        self._q3 = None if settings['_q3'] is None else pd.read_json(settings['_q3'])
        self.dummies = settings['dummies'] if 'dummies' in settings else {}
        self.is_fitted = True

    def infer_data_types(self, data=None):
        """
        In case no data types are provided, this function infers the most likely data types
        """
        if len(self.cat_cols) == len(self.num_cols) == len(self.date_cols) == 0:
            # First cleanup
            self.data = self.data.infer_objects() if data is None else data.infer_objects()

            # Iterate through keys
            for key in self.data.keys():
                # Skip target
                if key == self.target:
                    continue

                # Integer
                if pd.api.types.is_integer_dtype(self.data[key]):
                    self.int_cols.append(key)

                # Float
                if pd.api.types.is_float_dtype(self.data[key]):
                    self.float_cols.append(key)

                # Datetime
                if pd.api.types.is_datetime64_any_dtype(self.data[key]):
                    self.date_cols.append(key)

                # Booleans
                if pd.api.types.is_bool_dtype(self.data[key]):
                    self.int_cols.append(key)

                # Strings / Objects
                if pd.api.types.is_object_dtype(self.data[key]):

                    # Check date
                    date = self.data[key].astype('str').apply(pd.to_datetime, errors='coerce')
                    if date.isna().sum() < 0.3 * len(self.data):
                        self.date_cols.append(key)
                        self.data[key] = date
                    else:

                        # Check numeric
                        numeric = pd.to_numeric(self.data[key], errors='coerce', downcast='integer')
                        if numeric.isna().sum() < len(self.data) * 0.3:
                            # Float
                            if pd.api.types.is_float_dtype(numeric):
                                self.float_cols.append(key)

                            # Integer
                            if pd.api.types.is_integer_dtype(numeric):
                                self.int_cols.append(key)

                            # Update data
                            self.data[key] = numeric

                        # Check categorical cardinality
                        elif self.data[key].nunique() < 100:
                            self.cat_cols.append(key)

                        # Else warn
                        else:
                            warnings.warn(f"[AutoML] Couldn't identify feature: {key}")

            # Set num cols for reverse compatibility
            self.num_cols = self.int_cols + self.float_cols

            # Print
            if self.verbosity > 0:
                print(f"[AutoML] Found {len(self.int_cols)} integer, {len(self.float_cols)} float, {len(self.cat_cols)} "
                      f"categorical and {len(self.date_cols)} datetime columns")

        return

    def convert_data_types(self, data: pd.DataFrame = None, fit_categorical: bool = True) -> pd.DataFrame:
        """
        Cleans up the data types of all columns.

        Parameters
        ----------
        data [pd.DataFrame]: Input data
        fit_categorical [bool]: Whether to fit the categorical encoder

        Returns
        -------
        data [pd.DataFrame]: Cleaned input data
        """
        # Set data
        if data is not None:
            self.data = data

        # Datetime columns
        for key in self.date_cols:
            self.data.loc[:, key] = pd.to_datetime(self.data[key], errors='coerce', infer_datetime_format=True, utc=True)

        # Integer columns
        for key in self.int_cols:
            self.data.loc[:, key] = pd.to_numeric(self.data[key], errors='coerce', downcast='integer')

        # Float columns
        for key in self.float_cols:
            self.data.loc[:, key] = pd.to_numeric(self.data[key], errors='coerce', downcast='float')

        # Categorical columns
        if fit_categorical:
            self.data = self._fit_cat_cols()
        else:
            assert self.is_fitted, ".convert_data_types() was called with fit_categorical=False, while categorical " \
                                   "encoder is not yet fitted."
            self.data = self._transform_cat_cols()

        # We need everything to become numeric, so all that is not mentioned will be handled as numeric
        all_cols = self.float_cols + self.int_cols + self.date_cols + self.cat_cols + [self.target]
        for key in self.data.keys():
            if key not in all_cols:
                self.data.loc[:, key] = pd.to_numeric(self.data[key], errors='coerce')

        return self.data

    def _fit_cat_cols(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Encoding categorical variables always needs a scheme. This fits the scheme.
        """
        if data is not None:
            self.data = data

        for key in self.cat_cols:
            dummies = pd.get_dummies(self.data[key], prefix=key, dummy_na=True)
            self.data = self.data.drop(key, axis=1).join(dummies)
            self.dummies[key] = dummies.keys().tolist()
        return self.data

    def _transform_cat_cols(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Converts categorical variables according to fitted scheme.
        """
        if data is not None:
            self.data = data

        for key in self.cat_cols:
            value = self.dummies[key]
            dummies = [i[len(key) + 1:] for i in value]
            self.data[value] = np.equal.outer(self.data[key].values, dummies) * 1
            self.data = self.data.drop(key, axis=1)
        return self.data

    def remove_duplicates(self, data: pd.DataFrame = None, rows: bool = True) -> pd.DataFrame:
        """
        Removes duplicate columns and rows.

        Parameters
        ----------
        data [pd.DataFrame]: Input data

        Returns
        -------
        data [pd.DataFrame]: Cleaned input data
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
        if self.verbosity > 0 or (self.removedDuplicateColumns != 0 or self.removedDuplicateRows != 0):
            print(f'[AutoML] Removed {self.removedDuplicateColumns} duplicate columns and {self.removedDuplicateRows} '
                  f'duplicate rows')

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
            self.data = self.data.drop(columns=self.data.columns[self.data.nunique() == 1])

        # Note
        self.removedConstantColumns = columns - len(self.data.keys())
        if self.verbosity > 0 or self.removedConstantColumns != 0:
            print(f'[AutoML] Removed {self.removedConstantColumns} constant columns.')

        return self.data

    def fit_outliers(self, data: pd.DataFrame = None):
        """
        Checks outliers
        """
        if data is not None:
            self.data = data

        # With quantiles
        if self.outlier_removal == 'quantiles':
            self._q1 = self.data[self.num_cols].quantile(0.25)
            self._q3 = self.data[self.num_cols].quantile(0.75)

        # By z-score
        elif self.outlier_removal == 'z-score':
            self._means = self.data[self.num_cols].mean(skipna=True, numeric_only=True)
            self._stds = self.data[self.num_cols].std(skipna=True, numeric_only=True)
            self._stds[self._stds == 0] = 1

    def remove_outliers(self, data: pd.DataFrame = None, fit: bool = True) -> pd.DataFrame:
        """
        Removes outliers
        """
        if data is not None:
            self.data = data

        # Check if needs fitting
        if fit:
            self.fit_outliers(self.data)
        else:
            assert self.is_fitted, ".remove_outliers() is called with fit=False, yet the object isn't fitted yet."

        # With Quantiles
        if self.outlier_removal == 'quantiles':
            self.removedOutliers = ((self.data[self.num_cols] > self._q3).sum().sum() +
                                    (self.data[self.num_cols] < self._q1).sum().sum()).tolist()
            self.data[self.num_cols] = self.data[self.num_cols].mask(self.data[self.num_cols] < self._q1)
            self.data[self.num_cols] = self.data[self.num_cols].mask(self.data[self.num_cols] > self._q3)

        # With z-score
        elif self.outlier_removal == 'z-score':
            z_score = abs((self.data[self.num_cols] - self._means) / self._stds)
            self.removedOutliers = (z_score > self.z_score_threshold).sum().sum().tolist()
            self.data[self.num_cols] = self.data[self.num_cols].mask(z_score > self.z_score_threshold)

        # With clipping
        elif self.outlier_removal == 'clip':
            self.removedOutliers = ((self.data[self.num_cols] > 1e12).sum().sum() +
                                    (self.data[self.num_cols] < -1e12).sum().sum()).tolist()
            self.data[self.num_cols] = self.data[self.num_cols].clip(lower=-1e12, upper=1e12)
        return self.data

    def remove_missing_values(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fills missing values (infinities and 'not a number's)
        """
        if data is not None:
            self.data = data

        # Replace infinities
        self.data = self.data.replace([np.inf, -np.inf], np.nan)

        # Note
        self.imputedMissingValues = self.data[self.num_cols].isna().sum().sum().tolist()
        if self.verbosity > 0 or self.imputedMissingValues != 0:
            print(f'[AutoML] Imputed {self.imputedMissingValues} missing values.')

        # Removes all rows with missing values
        if self.missing_values == 'remove_rows':
            self.data = self.data[self.data.isna().sum(axis=1) == 0]

        # Removes all columns with missing values
        elif self.missing_values == 'remove_cols':
            self.data = self.data.loc[:, self.data.isna().sum(axis=0) == 0]

        # Fills all missing values with zero
        elif self.missing_values == 'zero':
            self.data = self.data.fillna(0)

        # Linearly interpolates missing values
        elif self.missing_values == 'interpolate':

            # Columns which are present with >10% missing values are not interpolated
            zero_keys = self.data.keys()[self.data.isna().sum() / len(self.data) > 0.1].tolist()

            # Get all non-date_cols & interpolate
            ik = np.setdiff1d(self.data.keys().to_list(), self.date_cols + zero_keys)
            self.data[ik] = self.data[ik].interpolate(limit_direction='both')

            # Fill date columns
            for key in self.date_cols:
                if self.data[key].isna().sum() != 0:
                    # Interpolate
                    ints = pd.Series(self.data[key].values.astype('int64'))
                    ints[ints < 0] = np.nan
                    self.data[key] = pd.to_datetime(ints.interpolate(), unit='ns')

                    # Uses date range (fixed interval)
                    # dr = pd.date_range(self.data[key].min(), self.data[key].max(), len(self.data))
                    # if (self.data[key] == dr).sum() > len(self.data) - self.data[key].isna().sum():
                    #     self.data[key] = dr

            # Fill rest (date & more missing values cols)
            if self.data.isna().sum().sum() != 0:
                self.data = self.data.fillna(0)

        # Fill missing values with column mean
        elif self.missing_values == 'mean':
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
                self.data[key] = pd.to_numeric(self.data[key], errors='coerce', downcast='integer')
        return self.data

    def clean_target(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Cleans the target column -- missing values already done, just converting classification classes
        """
        if data is not None:
            self.data = data

        if self.target in self.data:
            # Object is for sure classification
            if self.data[self.target].dtype == object:
                self.data[self.target] = self.data[self.target].astype('category').cat.codes

            # Classification check
            elif self.data[self.target].nunique() <= 0.5 * len(self.data):
                if sorted(set(self.data[self.target])) != [i for i in range(self.data[self.target].nunique())]:
                    for i, val in enumerate(sorted(set(self.data[self.target]))):
                        self.data.loc[self.data[self.target] == val, self.target] = i
        return self.data

    def _impute_columns(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        *** For production ***
        If a dataset is missing certain columns, this function looks at all registered columns and fills them with
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
            warnings.warn(f'Imputed {len(imputed)} missing columns! {imputed}')
        return self.data

    def prune_features(self, features: list):
        """
        For use with AutoML.Pipeline. We practically never use all features. Yet this processor imputes any missing
        features. This causes redundant operations, memory, and warnings. This function prunes the features to avoid
        that.

        parameters
        ----------
        features [list]: List with required features (NOTE: include required features for extracted)
        """
        hash_features = dict([(k, 0) for k in features])
        self.date_cols = [f for f in self.date_cols if f in hash_features]
        self.num_cols = [f for f in self.num_cols if f in hash_features]
        self.int_cols = [f for f in self.int_cols if f in hash_features]
        self.float_cols = [f for f in self.float_cols if f in hash_features]
        self.cat_cols = [f for f in self.cat_cols if f in hash_features]
