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
                 include_output: bool = False,
                 missing_values: str = 'interpolate',
                 outlier_removal: str = 'clip',
                 z_score_threshold: int = 4,
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
        num_cols list: Numerical columns, all parsed to integers and floats
        date_cols list: Date columns, all parsed to pd.datetime format
        cat_cols list: Categorical Columns. Currently all one-hot encoded.
        missing_values str: How to deal with missing values ('remove', 'interpolate' or 'mean')
        outlier_removal str: How to deal with outliers ('clip', 'quantiles', 'z-score' or 'none')
        z_score_threshold int: If outlierRemoval='z-score', the threshold is adaptable, default=4.
        folder str: Directory for storing the output files
        version int: Versioning the output files
        mode str: classification / regression
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

        # Fitted Settings
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
        data = clean_keys(data)

        # Remove Duplicates
        data = self.remove_duplicates(data)

        # Infer data-types
        self.infer_data_types(data)

        # Convert data types
        data = self.convert_data_types(data, fit_categorical=True)

        # Remove outliers
        data = self.remove_outliers(data, fit=True)

        # Remove missing values
        data = self.remove_missing_values(data)

        # Remove Constants
        data = self.remove_constants(data)

        # Convert integer columns
        data = self.convert_float_int(data)

        # Clean target
        data = self.clean_target(data)

        # Finish
        self.is_fitted = True
        if self.verbosity > 0:
            print('[AutoML] Processing completed, ({} x {}) samples returned'.format(len(data), len(data.keys())))
        return data

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
        data = clean_keys(data)

        # Impute columns
        data = self._impute_columns(data)

        # Remove duplicates
        data = self.remove_duplicates(data, rows=False)

        # Convert data types
        data = self.convert_data_types(data, fit_categorical=False)

        # Remove outliers
        data = self.remove_outliers(data, fit=False)

        # Remove missing values
        data = self.remove_missing_values(data)

        # Convert integer columns
        data = self.convert_float_int(data)

        return data

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

    def infer_data_types(self, data: pd.DataFrame):
        """
        In case no data types are provided, this function infers the most likely data types
        """
        if len(self.cat_cols) == len(self.num_cols) == len(self.date_cols) == 0:
            # First cleanup
            data = data.infer_objects()

            # Remove target from columns
            if not self.includeOutput and self.target is not None and self.target in data:
                data = data.drop(self.target, axis=1)

            # Iterate through keys
            for key in data.keys():
                # Integer
                if pd.api.types.is_integer_dtype(data[key]):
                    self.int_cols.append(key)

                # Float
                if pd.api.types.is_float_dtype(data[key]):
                    self.float_cols.append(key)

                # Datetime
                if pd.api.types.is_datetime64_any_dtype(data[key]):
                    self.date_cols.append(key)

                # Booleans
                if pd.api.types.is_bool_dtype(data[key]):
                    self.int_cols.append(key)

                # Strings / Objects
                if pd.api.types.is_object_dtype(data[key]):
                    is_date = data[key].astype('str').apply(pd.to_datetime, errors='coerce').isna().sum() < 0.3 * \
                              len(data)
                    if is_date:
                        self.date_cols.append(key)
                    elif data[key].nunique() < 100:
                        self.cat_cols.append(key)
                    else:
                        forced_numeric = pd.to_numeric(data[key], errors='coerce')
                        if forced_numeric.isna().sum() < len(data) / 25:
                            self.float_cols.append(key)

            # Set num cols for reverse compatibility
            self.num_cols = self.int_cols + self.float_cols

            # Check if float keys are secretly not integers
            for key in self.float_cols:
                forced_int = pd.to_numeric(data[key].fillna(0), errors='coerce', downcast='integer')
                if pd.api.types.is_integer_dtype(forced_int):
                    self.float_cols.remove(key)
                    self.int_cols.append(key)

            # Print
            if self.verbosity > 0:
                print(f"[AutoML] Found {len(self.int_cols)} integer, {len(self.float_cols)} float, {len(self.cat_cols)} "
                      f"categorical and {len(self.date_cols)} datetime columns")

        return

    def convert_data_types(self, data: pd.DataFrame, fit_categorical: bool = True) -> pd.DataFrame:
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
        # Datetime columns
        for key in self.date_cols:
            data.loc[:, key] = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True, utc=True)

        # Integer columns
        for key in self.int_cols:
            data.loc[:, key] = pd.to_numeric(data[key], errors='coerce', downcast='integer')

        # Float columns
        for key in self.float_cols:
            data.loc[:, key] = pd.to_numeric(data[key], errors='coerce', downcast='float')

        # Categorical columns
        if fit_categorical:
            data = self._fit_cat_cols(data)
        else:
            assert self.is_fitted, ".convert_data_types() was called with fit_categorical=False, while categorical " \
                                   "encoder is not yet fitted."
            data = self._transform_cat_cols(data)

        # We need everything to become numeric, so all that is not mentioned will be handled as numeric
        all_cols = self.float_cols + self.int_cols + self.date_cols + self.cat_cols + [self.target]
        for key in data.keys():
            if key not in all_cols:
                data.loc[:, key] = pd.to_numeric(data[key], errors='coerce')

        return data

    def _fit_cat_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encoding categorical variables always needs a scheme. This fits the scheme.
        """
        for key in self.cat_cols:
            dummies = pd.get_dummies(data[key], prefix=key, drop_first=True)
            data = data.drop(key, axis=1).join(dummies)
            self.dummies[key] = dummies.keys().tolist()
        return data

    def _transform_cat_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts categorical variables according to fitted scheme.
        """
        for key in self.cat_cols:
            value = self.dummies[key]
            dummies = [i[len(key) + 1:] for i in value]
            data[value] = np.equal.outer(data[key].values, dummies) * 1
            data = data.drop(key, axis=1)
        return data

    def remove_duplicates(self, data: pd.DataFrame, rows: bool = True) -> pd.DataFrame:
        """
        Removes duplicate columns and rows.

        Parameters
        ----------
        data [pd.DataFrame]: Input data

        Returns
        -------
        data [pd.DataFrame]: Cleaned input data
        """
        # Note down
        n_rows, n_columns = len(data), len(data.keys())

        # Remove Duplicates
        if rows:
            data = data.drop_duplicates()
        data = data.loc[:, ~data.columns.duplicated()]

        # Note
        self.removedDuplicateColumns = n_columns - len(data.keys())
        self.removedDuplicateRows = n_rows - len(data)
        if self.verbosity > 0 or (self.removedDuplicateColumns != 0 or self.removedDuplicateRows != 0):
            print(f'[AutoML] Removed {self.removedDuplicateColumns} duplicate columns and {self.removedDuplicateRows} '
                  f'duplicate rows')

        return data

    def remove_constants(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes constant columns
        """
        columns = len(data.keys())

        # Remove Constants
        data = data.drop(columns=data.columns[data.nunique() == 1])

        # Note
        self.removedConstantColumns = columns - len(data.keys())
        if self.verbosity > 0 or self.removedConstantColumns != 0:
            print(f'[AutoML] Removed {self.removedConstantColumns} constant columns.')

        return data

    def fit_outliers(self, data: pd.DataFrame) -> int:
        """
        Checks outliers
        """
        # With quantiles
        if self.outlier_removal == 'quantiles':
            self._q1 = data[self.num_cols].quantile(0.25)
            self._q3 = data[self.num_cols].quantile(0.75)

        # By z-score
        elif self.outlier_removal == 'z-score':
            self._means = data[self.num_cols].mean(skipna=True, numeric_only=True)
            self._stds = data[self.num_cols].std(skipna=True, numeric_only=True)
            self._stds[self._stds == 0] = 1

    def remove_outliers(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Removes outliers
        """
        # Check if needs fitting
        if fit:
            self.fit_outliers(data)
        else:
            assert self.is_fitted, ".remove_outliers() is called with fit=False, yet the object isn't fitted yet."

        # With Quantiles
        if self.outlier_removal == 'quantiles':
            self.removedOutliers = ((data[self.num_cols] > self._q3).sum().sum() +
                                    (data[self.num_cols] < self._q1).sum().sum()).tolist()
            data[self.num_cols] = data[self.num_cols].mask(data[self.num_cols] < self._q1)
            data[self.num_cols] = data[self.num_cols].mask(data[self.num_cols] > self._q3)

        # With z-score
        elif self.outlier_removal == 'z-score':
            z_score = abs((data[self.num_cols] - self._means) / self._stds)
            self.removedOutliers = (z_score > self.z_score_threshold).sum().sum().tolist()
            data[self.num_cols] = data[self.num_cols].mask(z_score > self.z_score_threshold)

        # With clipping
        elif self.outlier_removal == 'clip':
            self.removedOutliers = ((data[self.num_cols] > 1e12).sum().sum() +
                                    (data[self.num_cols] < -1e12).sum().sum()).tolist()
            data[self.num_cols] = data[self.num_cols].clip(lower=-1e12, upper=1e12)
        return data

    def remove_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values (infinities and 'not a number's)
        """
        # Replace infinities
        data = data.replace([np.inf, -np.inf], np.nan)

        # Note
        self.imputedMissingValues = data[self.num_cols].isna().sum().sum().tolist()
        if self.verbosity > 0 or self.imputedMissingValues != 0:
            print(f'[AutoML] Imputed {self.imputedMissingValues} missing values.')

        # Removes all rows with missing values
        if self.missing_values == 'remove_rows':
            data = data[data.isna().sum(axis=1) == 0]

        # Removes all columns with missing values
        elif self.missing_values == 'remove_cols':
            data = data.loc[:, data.isna().sum(axis=0) == 0]

        # Fills all missing values with zero
        elif self.missing_values == 'zero':
            data = data.fillna(0)

        # Linearly interpolates missing values
        elif self.missing_values == 'interpolate':

            # Columns which are present with >10% missing values are not interpolated
            zero_keys = data.keys()[data.isna().sum() / len(data) > 0.1].tolist()

            # Get all non-date_cols & interpolate
            ik = np.setdiff1d(data.keys().to_list(), self.date_cols + zero_keys)
            data[ik] = data[ik].interpolate(limit_direction='both')

            # Fill date columns
            for key in self.date_cols:
                if data[key].isna().sum() != 0:
                    # Interpolate
                    ints = pd.Series(data[key].values.astype('int64'))
                    ints[ints < 0] = np.nan
                    data[key] = pd.to_datetime(ints.interpolate(), unit='ns')

                    # Uses date range (fixed interval)
                    # dr = pd.date_range(data[key].min(), data[key].max(), len(data))
                    # if (data[key] == dr).sum() > len(data) - data[key].isna().sum():
                    #     data[key] = dr

            # Fill rest (date & more missing values cols)
            if data.isna().sum().sum() != 0:
                data = data.fillna(0)

        # Fill missing values with column mean
        elif self.missing_values == 'mean':
            data = data.fillna(data.mean())

            # Need to be individual for some reason
            for key in self.date_cols:
                data[key] = data[key].fillna(data[key].mean())

        return data

    def convert_float_int(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Integer columns with NaN in them are interpreted as floats.
        In the beginning we check whether some columns should be integers,
        but we rely on different algorithms to take care of the NaN.
        Therefore, we only convert floats to integers at the very end
        """
        for key in self.int_cols:
            if key in data:
                data[key] = pd.to_numeric(data[key], errors='coerce', downcast='integer')
        return data

    def clean_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the target column -- missing values already done, just converting classification classes
        """
        if self.target in data:
            # Object is for sure classification
            if data[self.target].dtype == object:
                data[self.target] = data[self.target].astype('category').cat.codes

            # Classification check
            elif data[self.target].nunique() <= 0.5 * len(data):
                if sorted(set(data[self.target])) != [i for i in range(data[self.target].nunique())]:
                    for i, val in enumerate(sorted(set(data[self.target]))):
                        data.loc[data[self.target] == val, self.target] = i
        return data

    def _impute_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        *** For production ***
        If a dataset is missing certain columns, this function looks at all registered columns and fills them with
        zeros.
        """
        imputed = []
        for keys in [self.num_cols, self.date_cols, self.cat_cols]:
            for key in [k for k in keys if k not in data]:
                data[key] = np.zeros(len(data))
                imputed.append(key)
        if len(imputed) > 0:
            warnings.warn(f'Imputed {len(imputed)} missing columns! {imputed}')
        return data

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
