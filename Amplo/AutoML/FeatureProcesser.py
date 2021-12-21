import time
import copy
import warnings
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from boruta import BorutaPy
from shap import TreeExplainer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans
from Amplo.Utils import clean_keys
from Amplo.Classifiers import CatBoostClassifier
from Amplo.Regressors import CatBoostRegressor


class FeatureProcesser:

    def __init__(self,
                 max_lags: int = 10,
                 max_diff: int = 2,
                 date_cols: list = None,
                 information_threshold: float = 0.99,
                 selection_cutoff: float = 0.85,
                 selection_increment: float = 0.005,
                 extract_features: bool = True,
                 mode: str = 'classification',
                 timeout: int = 900,
                 verbosity: int = 1):
        """
        Automatically extracts and selects features. Removes Co-Linear Features.
        Included Feature Extraction algorithms:
        - Multiplicative Features
        - Dividing Features
        - Additive Features
        - Subtractive Features
        - Trigonometric Features
        - K-Means Features
        - Lagged Features
        - Differencing Features

        Included Feature Selection algorithms:
        - Random Forest Feature Importance (Threshold and Increment)
        - Predictive Power Score
        - Boruta

        Parameters
        ----------
        max_lags [int]: Maximum lags for lagged features to analyse
        max_diff [int]: Maximum differencing order for differencing features
        date_cols [list[str]]: List of datetime columns, for datetime features
        information_threshold [float]: Information threshold for co-linear features
        extract_features [bool]: Whether or not to extract features
        folder [str]: Parent folder for results
        mode [str]: classification / regression
        timeout [int]: Feature Extraction can be exhausting for many features, this limits the scope
        version [int]: To version all stored files
        """
        # Tests
        assert 0 <= max_lags < 120, 'Max lags needs to be within [0, 50]'
        assert 0 <= max_diff < 3, 'Max diff needs to be within [0, 3]'
        assert 0 < information_threshold < 1, 'Information threshold needs to be within [0, 1]'
        assert mode.lower() in ['classification', 'regression'], \
            'Mode needs to be specified (regression or classification)'

        # Parameters
        self.x = None
        self.originalInput = None
        self.y = None
        self.model = None
        self.mode = mode
        self.timeout = timeout
        self.date_cols = [] if date_cols is None else date_cols

        # Register
        self.is_fitted = False
        self.baseScore = {}
        self.coLinearFeatures = []
        self.linearFeatures = []
        self.crossFeatures = []
        self.trigonometricFeatures = []
        self.inverseFeatures = []
        self.kMeansFeatures = []
        self.laggedFeatures = []
        self.diffFeatures = []
        self.datetimeFeatures = []
        self.featureSets = {}
        self._means = None
        self._stds = None
        self._centers = None
        self.featureImportance = {}

        # Parameters
        self.maxLags = max_lags
        self.maxDiff = max_diff
        self.informationThreshold = information_threshold
        self.selectionCutoff = selection_cutoff
        self.selectionIncrement = selection_increment
        self.extractFeatures = extract_features
        self.verbosity = verbosity

    def fit_transform(self, x: pd.DataFrame, y: pd.Series) -> [pd.DataFrame, dict]:
        """
        Extracts features, and selects them
        Parameters
        ----------
        x [pd.DataFrame]: Input data (features)
        y [pd.Series]: Output data (dependent)

        Returns
        -------
        x [pd.DataFrame]: Input data with enriched features
        feature_sets [dict]: A dictionary with various selected feature sets
        """
        # First clean (just to make sure), and set in class memory
        self._clean_set(x, y)

        # Remove co-linear features
        self._remove_co_linearity()

        # Extract
        if self.extractFeatures:
            self._calc_baseline()

            self._fit_datetime_features()
            self._add_datetime_features()

            self._fit_cross_features()
            self._add_cross_features()

            self._fit_k_means_features()
            self._add_k_means_features()

            self._fit_trigonometry_features()
            self._add_trigonometry_features()

            self._fit_inverse_features()
            self._add_inverse_features()

            self._fit_diff_features()
            self._add_diff_features()

            self._fit_lagged_features()
            self._add_lagged_features()

        # Select
        # self.featureSets['PPS'] = self._sel_predictive_power_score()
        self.featureSets['RFT'], self.featureSets['RFI'] = self._sel_gini_impurity()
        self.featureSets['ShapThreshold'], self.featureSets['ShapIncrement'] = self._sel_shap()

        # Set fitted
        self.is_fitted = True

        return self.x, self.featureSets

    def transform(self, data: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        """
        Transforms dataset features into enriched and selected features

        Parameters
        ----------
        data [pd.DataFrame]: Input data
        feature_set [str]: Which feature set to use for selection
        """
        assert self.is_fitted, "Can only use transform after object is fitted."

        # Clean & downcast data
        data = data.astype('float32').clip(lower=-1e12, upper=1e12).fillna(0)

        # Get original features
        features = self.featureSets[feature_set]
        required = self.get_required_features(features)

        # Impute missing keys
        missing_keys = [k for k in required if k not in data.keys()]
        if len(missing_keys) > 0:
            warnings.warn('[AutoML] Imputing {} keys: {}'.format(len(missing_keys), missing_keys))
        for k in missing_keys:
            data.loc[:, k] = np.zeros(len(data))

        # Set output
        self.x = data

        # Linear features
        self._add_linear_features()

        # Cross features
        self._add_cross_features()

        # Datetime features
        self._add_datetime_features()

        # Differentiated features
        self._add_diff_features()

        # K-Means features
        self._add_k_means_features()

        # Lagged features
        self._add_lagged_features()

        # Trigonometric features
        self._add_trigonometry_features()

        # Inverse Features
        self._add_inverse_features()

        # Enforce the right order of features
        self.x = self.x[features]

        # And clip everything (we do this with all features in ._analyse_feature(), no exception)
        self.x = self.x.astype('float32').clip(lower=-1e12, upper=1e12).fillna(0)

        return self.x

    def get_settings(self) -> dict:
        """
        Get settings to recreate fitted object.
        """
        assert self.is_fitted, "Object not yet fitted."
        return {
            'linearFeatures': self.linearFeatures,
            'crossFeatures': self.crossFeatures,
            'trigonometricFeatures': self.trigonometricFeatures,
            'inverseFeatures': self.inverseFeatures,
            'kMeansFeatures': self.kMeansFeatures,
            'laggedFeatures': self.laggedFeatures,
            'diffFeatures': self.diffFeatures,
            'featureSets': self.featureSets,
            '_means': '[]' if self._means is None else self._means.to_json(),
            '_stds': '[]' if self._stds is None else self._stds.to_json(),
            '_centers': '[]' if self._centers is None else self._centers.to_json(),
            'featureImportance': self.featureImportance
        }

    def load_settings(self, settings: dict) -> None:
        """
        Loads settings from dictionary and recreates a fitted object
        """
        self.linearFeatures = settings['linearFeatures'] if 'linearFeatures' in settings else []
        self.crossFeatures = settings['crossFeatures'] if 'crossFeatures' in settings else []
        self.trigonometricFeatures = settings['trigonometricFeatures'] if 'trigonometricFeatures' in settings else []
        self.inverseFeatures = settings['inverseFeatures'] if 'inverseFeatures' in settings else []
        self.kMeansFeatures = settings['kMeansFeatures'] if 'kMeansFeatures' in settings else []
        self.laggedFeatures = settings['laggedFeatures'] if 'laggedFeatures' in settings else []
        self.diffFeatures = settings['diffFeatures'] if 'diffFeatures' in settings else {}
        self.featureSets = settings['featureSets'] if 'featureSets' in settings else []
        self._means = pd.read_json(settings['_means'], typ='series') if 'v' in settings else []
        self._stds = pd.read_json(settings['_stds'], typ='series') if '_stds' in settings else []
        self._centers = pd.read_json(settings['_centers']) if '_centers' in settings else []
        self.featureImportance = settings['featureImportance'] if 'featureImportance' in settings else {}
        self.is_fitted = True
        self.verbosity = 0

    def _clean_set(self, x: pd.DataFrame, y: pd.Series) -> None:
        """
        Does some basic cleaning just in case, and sets the input/output in memory.
        """
        # Sets validation model
        if self.mode == 'classification':
            self.model = DecisionTreeClassifier(max_depth=3)
            if y.nunique() > 2:
                self.mode = 'multiclass'
        elif self.mode == 'regression':
            self.model = DecisionTreeRegressor(max_depth=3)

        # Bit of necessary data cleaning (shouldn't change anything)
        x = clean_keys(x.astype('float32').clip(lower=-1e12, upper=1e12).fillna(0).reset_index(drop=True))
        self.x = copy.copy(x)
        self.originalInput = copy.copy(x)
        self.y = y.replace([np.inf, -np.inf], 0).fillna(0).reset_index(drop=True)

    def _calc_baseline(self):
        """
        Calculates feature value of the original features
        """
        baseline = {}
        for key in self.originalInput.keys():
            baseline[key] = self._analyse_feature(self.x[key])
        self.baseline = pd.DataFrame(baseline).max(axis=1)

    def _analyse_feature(self, feature: pd.Series) -> list:
        """
        Analyses and scores a feature
        In case of multiclass, we score per class :)
        """
        # Clean feature
        feature = feature.clip(lower=-1e12, upper=1e12).fillna(0).values.reshape((-1, 1))

        # Copy & fit model
        m = copy.copy(self.model)
        m.fit(feature, self.y)

        # Score
        if self.mode == 'multiclass':
            # With weight (self.y == i)
            return [m.score(feature, self.y, self.y == i) for i in self.y.unique()]
        else:
            # Without weight
            return [m.score(feature, self.y)]

    def _accept_feature(self, score: list) -> bool:
        """
        Whether or not to accept a new feature,
        basically we accept if it's higher than baseline for any of the classes
        """
        if any(score > self.baseline.values):
            return True
        else:
            return False

    @ staticmethod
    def _select_features(scores: dict) -> list:
        """
        Run at the end of all feature extraction calls.
        Select best 50 features per class.
        In case of regression, just best 50 (only one 'class')

        Parameters
        ----------
        scores [dict]: Features as keys, scores as values

        Returns
        -------
        selected_features [list[str]]: Returns all selected features
        """
        # If scores is empty, return empty list
        if len(scores) == 0:
            return []

        # Convert dict to dataframe
        scores = pd.DataFrame(scores)

        # Select indices
        features_per_class = 50
        indices = []
        for score in range(len(scores)):    # Loop through classes
            indices += [i for i, k in enumerate(scores.keys())
                        if k in scores.loc[score].sort_values(ascending=False).keys()[:features_per_class]]

        # Return Keys
        return list(scores.keys()[np.unique(indices)])

    def _remove_co_linearity(self):
        """
        Calculates the Pearson Correlation Coefficient for all input features.
        Those higher than the information threshold are linearly codependent (i.e., describable by y = a x + b)
        These features add little to no information and are therefore removed.
        """
        if self.verbosity > 0:
            print('[AutoML] Analysing co-linearity')

        # Get co-linear features
        nk = len(self.x.keys())
        norm = (self.x - self.x.mean(skipna=True, numeric_only=True)).to_numpy()
        ss = np.sqrt(np.sum(norm ** 2, axis=0))
        corr_mat = np.zeros((nk, nk))
        for i in range(nk):
            for j in range(nk):
                if i == j:
                    continue
                if corr_mat[i, j] == 0:
                    c = abs(np.sum(norm[:, i] * norm[:, j]) / (ss[i] * ss[j]))
                    corr_mat[i, j] = c
                    corr_mat[j, i] = c
        upper = np.triu(corr_mat)
        self.coLinearFeatures = self.x.keys()[np.sum(upper > self.informationThreshold, axis=0) > 0].to_list()

        # Parse results
        self.originalInput = self.originalInput.drop(self.coLinearFeatures, axis=1)
        if self.verbosity > 0:
            print('[AutoML] Removed {} Co-Linear features ({:.3f} %% threshold)'
                  .format(len(self.coLinearFeatures), self.informationThreshold))

    # Start Feature functions
    def _fit_datetime_features(self):
        """
        Finds interesting datetime features
        """
        if self.verbosity > 0:
            print('[AutoML] Analysing datetime features')
        scores = {}

        # Analyse datetime features
        for key in self.date_cols:
            scores[f"{key}__dt__dayofyear"] = self.x[key].dt.dayofyear
            scores[f"{key}__dt__dayofweek"] = self.x[key].dt.dayofweek
            scores[f"{key}__dt__quarter"] = self.x[key].dt.quarter
            scores[f"{key}__dt__month"] = self.x[key].dt.month
            scores[f"{key}__dt__week"] = self.x[key].dt.week
            scores[f"{key}__dt__hour"] = self.x[key].dt.hour
            scores[f"{key}__dt__minute"] = self.x[key].dt.minute
            scores[f"{key}__dt__second"] = self.x[key].dt.second

        # Select valuable features
        self.datetimeFeatures = self._select_features(scores)

    def _add_datetime_features(self):
        """
        Calculates various datetime features.
        """
        # Add features
        for k in self.datetimeFeatures:
            key, _, period = k.split('__')
            self.x[k] = getattr(self.x[key].dt, period).clip(lower=-1e12, upper=1e12).fillna(0)

        # Print result
        if self.verbosity > 0:
            print('[AutoML] Added {} datetime features'.format(len(self.datetimeFeatures)))

    def _fit_cross_features(self):
        """
        Analyses cross-features --> division and multiplication
        Should be limited to say ~500.000 features (runs about 100-150 features / second)
        """
        if self.verbosity > 0:
            print('[AutoML] Analysing cross features')
        scores = {}
        n_keys = len(self.originalInput.keys())
        start_time = time.time()

        # Analyse Cross Features
        for i, key_a in enumerate(tqdm(self.originalInput.keys())):
            accepted_for_key_a = 0
            for j, key_b in enumerate(self.originalInput.keys()[np.random.permutation(n_keys)]):
                # Skip if they're the same
                if key_a == key_b:
                    continue
                # Skip rest if key_a is not useful in first max(50, 30%) (uniform)
                if accepted_for_key_a == 0 and j > max(50, int(n_keys / 3)):
                    continue
                # Skip if we're out of time
                if time.time() - start_time > self.timeout:
                    continue

                # Analyse Division
                feature = self.x[key_a] / self.x[key_b]
                score = self._analyse_feature(feature)
                # Accept or not
                if self._accept_feature(score):
                    scores[key_a + '__d__' + key_b] = score
                    accepted_for_key_a += 1

                # Multiplication i * j == j * i, so skip if j >= i
                if j > i:
                    continue

                # Analyse Multiplication
                feature = self.x[key_a] * self.x[key_b]
                score = self._analyse_feature(feature)
                # Accept or not
                if self._accept_feature(score):
                    scores[key_a + '__x__' + key_b] = score
                    accepted_for_key_a += 1

        # Select valuable features
        self.crossFeatures = self._select_features(scores)

    def _add_cross_features(self):
        """
        Adds all selected cross features
        """
        # Add features
        for k in self.crossFeatures:
            if '__x__' in k:
                key_a, key_b = k.split('__x__')
                feature = self.x[key_a] * self.x[key_b]
                self.x[k] = feature.clip(lower=-1e12, upper=1e12).fillna(0)
            else:
                key_a, key_b = k.split('__d__')
                feature = self.x[key_a] / self.x[key_b]
                self.x[k] = feature.clip(lower=-1e12, upper=1e12).fillna(0)

        # Print result
        if self.verbosity > 0:
            print('[AutoML] Added {} cross features'.format(len(self.crossFeatures)))

    def _fit_trigonometry_features(self):
        """
        Calculates trigonometry features with sinus, cosines
        """
        if self.verbosity > 0:
            print('[AutoML] Analysing Trigonometric Features')

        scores = {}
        for key in tqdm(self.originalInput.keys()):
            # Sinus feature
            sin_feature = np.sin(self.x[key])
            score = self._analyse_feature(sin_feature)
            if self._accept_feature(score):
                scores['sin__' + key] = score

            # Co sinus feature
            cos_feature = np.cos(self.x[key])
            score = self._analyse_feature(cos_feature)
            if self._accept_feature(score):
                scores['cos__' + key] = score

        # Select valuable features
        self.trigonometricFeatures = self._select_features(scores)

    def _add_trigonometry_features(self):
        """
        Adds selected trigonometry features with sinus, cosines
        """
        # Add features
        for k in self.trigonometricFeatures:
            func, key = k.split('__')
            self.x[k] = getattr(np, func)(self.x[key])

        # Store
        if self.verbosity > 0:
            print('[AutoML] Added {} trigonometric features'.format(len(self.trigonometricFeatures)))

    def _fit_linear_features(self):
        """
        Analyses simple additive and subtractive features
        """
        # Load if available
        if self.verbosity > 0:
            print('[AutoML] Analysing Linear Features'.format(len(self.linearFeatures)))

        scores = {}
        start_time = time.time()
        n_keys = len(self.originalInput.keys())
        for i, key_a in enumerate(self.originalInput.keys()):
            accepted_for_key_a = 0
            for j, key_b in enumerate(self.originalInput.keys()[np.random.permutation(n_keys)]):
                # Skip if they're the same
                if key_a == key_b:
                    continue
                # Skip rest if key_a is not useful in first max(50, 30%) (uniform)
                if accepted_for_key_a == 0 and j > max(50, int(n_keys / 3)):
                    continue
                # Skip if we're out of time
                if time.time() - start_time > self.timeout:
                    continue

                # Subtracting feature
                feature = self.x[key_a] - self.x[key_b]
                score = self._analyse_feature(feature)
                if self._accept_feature(score):
                    scores[key_a + '__sub__' + key_b] = score
                    accepted_for_key_a += 1

                # A + B == B + A, so skip if i > j
                if j > i:
                    continue

                # Additive feature
                feature = self.x[key_a] + self.x[key_b]
                score = self._analyse_feature(feature)
                if self._accept_feature(score):
                    scores[key_a + '__add__' + key_b] = score
                    accepted_for_key_a += 1

        # Select valuable Features
        self.linearFeatures = self._select_features(scores)

    def _add_linear_features(self):
        """
        Adds selected additive and subtractive features
        """
        # Add features
        for key in self.linearFeatures:
            if '__sub__' in key:
                key_a, key_b = key.split('__sub__')
                feature = self.x[key_a] - self.x[key_b]
                self.x.loc[:, key] = feature.clip(lower=-1e12, upper=1e12).fillna(0)
            else:
                key_a, key_b = key.split('__add__')
                feature = self.x[key_a] + self.x[key_b]
                self.x.loc[:, key] = feature.clip(lower=-1e12, upper=1e12).fillna(0)

        # store
        if self.verbosity > 0:
            print('[AutoML] Added {} additive features'.format(len(self.linearFeatures)))

    def _fit_inverse_features(self):
        """
        Analyses inverse features.
        """
        if self.verbosity > 0:
            print('[AutoML] Analysing Inverse Features')

        scores = {}
        for i, key in enumerate(self.originalInput.keys()):
            inv_feature = (1 / self.x[key]).clip(lower=-1e12, upper=1e12)
            score = self._analyse_feature(inv_feature)
            if self._accept_feature(score):
                scores['inv__' + key] = score

        self.inverseFeatures = self._select_features(scores)

    def _add_inverse_features(self):
        """
        Fits inverse features.
        """
        # Add features
        for k in self.inverseFeatures:
            key = k[5:]
            self.x[k] = (1 / self.x[key]).clip(lower=-1e12, upper=1e12)

        # Store
        if self.verbosity > 0:
            print('[AutoML] Added {} inverse features.'.format(len(self.inverseFeatures)))

    def _fit_k_means_features(self):
        """
        Analyses the correlation of k-means features.
        k-means is a clustering algorithm which clusters the data.
        The distance to each cluster is then analysed.
        """
        # Check if not exist
        if self.verbosity > 0:
            print('[AutoML] Calculating and Analysing K-Means features')

        # Prepare data
        data = copy.copy(self.originalInput)
        self._means = data.mean()
        self._stds = data.std()
        self._stds[self._stds == 0] = 1
        data -= self._means
        data /= self._stds

        # Determine clusters
        clusters = min(max(int(np.log10(len(self.originalInput)) * 8), 8), len(self.originalInput.keys()))
        k_means = MiniBatchKMeans(n_clusters=clusters)
        column_names = ['dist__{}_{}'.format(i, clusters) for i in range(clusters)]
        distances = pd.DataFrame(columns=column_names, data=k_means.fit_transform(data))
        distances = distances.clip(lower=-1e12, upper=1e12).fillna(0)
        self._centers = pd.DataFrame(columns=self.originalInput.keys(), data=k_means.cluster_centers_)

        # Analyse correlation
        scores = {}
        for key in tqdm(distances.keys()):
            score = self._analyse_feature(distances[key])
            if self._accept_feature(score):
                scores[key] = score

        # Add the valuable features
        self.kMeansFeatures = self._select_features(scores)

    def _add_k_means_features(self):
        """
        Adds the selected k-means features
        """
        # Only execute if necessary
        if len(self.kMeansFeatures) > 0:
            # Normalize data
            tmp = copy.deepcopy(self.x.loc[:, self._centers.keys()])
            tmp -= self._means
            tmp /= self._stds

            # Calculate centers
            for key in self.kMeansFeatures:
                ind = int(key[key.find('dist__') + 6: key.rfind('_')])
                self.x.loc[:, key] = np.sqrt(np.square(tmp - self._centers.iloc[ind]).sum(axis=1))

            if self.verbosity > 0:
                print('[AutoML] Added {} K-Means features'.format(len(self.kMeansFeatures)))

    def _fit_diff_features(self):
        """
        Analyses whether the diff signal of the data should be included.
        """
        # Check if we're allowed
        if self.maxDiff == 0:
            if self.verbosity > 0:
                print('[AutoML] Diff features skipped, max diff = 0')
            return

        if self.verbosity > 0:
            print('[AutoML] Analysing diff features')

        # Copy data so we can diff without altering original data
        keys = self.originalInput.keys()
        diff_input = copy.copy(self.originalInput)

        # Calculate scores
        scores = {}
        for diff in tqdm(range(1, self.maxDiff + 1)):
            diff_input = diff_input.diff().fillna(0)
            for key in keys:
                score = self._analyse_feature(diff_input[key])
                if self._accept_feature(score):
                    scores[key + '__diff__{}'.format(diff)] = score

        # Select the valuable features
        self.diffFeatures = self._select_features(scores)

    def _add_diff_features(self):
        """
        Adds whether the diff signal of the data should be included.
        """
        # Add Diff Features
        for k in self.diffFeatures:
            key, diff = k.split('__diff__')
            feature = self.x[key]
            for i in range(1, int(diff)):
                feature = feature.diff().clip(lower=-1e12, upper=1e12).fillna(0)
            self.x[k] = feature

        # Print output
        if self.verbosity > 0:
            print('[AutoML] Added {} differenced features'.format(len(self.diffFeatures)))

    def _fit_lagged_features(self):
        """
        Analyses the correlation of lagged features (value of sensor_x at t-1 to predict target at t)
        """
        # Check if allowed
        if self.maxLags == 0:
            if self.verbosity > 0:
                print('[AutoML] Lagged features skipped, max lags = 0')
            return

        if self.verbosity > 0:
            print('[AutoML] Analysing lagged features')

        # Analyse
        keys = self.originalInput.keys()
        scores = {}
        for lag in tqdm(range(1, self.maxLags)):
            for key in keys:
                score = self._analyse_feature(self.x[key].shift(lag))
                if self._accept_feature(score):
                    scores[key + '__lag__{}'.format(lag)] = score

        # Select
        self.laggedFeatures = self._select_features(scores)

    def _add_lagged_features(self):
        """
        Adds the correlation of lagged features (value of sensor_x at t-1 to predict target at t)
        """
        # Add selected
        for k in self.laggedFeatures:
            key, lag = k.split('__lag__')
            self.x[k] = self.x[key].shift(-int(lag), fill_value=0)

        if self.verbosity > 0:
            print('[AutoML] Added {} lagged features'.format(len(self.laggedFeatures)))

    # def _sel_predictive_power_score(self):
    #     """
    #     Calculates the Predictive Power Score (https://github.com/8080labs/ppscore)
    #     Asymmetric correlation based on single decision trees trained on 5.000 samples with 4-Fold validation.
    #     """
    #     if self.verbosity > 0:
    #         print('[AutoML] Determining features with PPS')
    #
    #     # Copy data
    #     data = self.x.copy()
    #     data['target'] = self.y.copy()
    #
    #     # Get Predictive Power Score
    #     pp_score = ppscore.predictors(data, "target")
    #
    #     # Select columns
    #     pp_cols = pp_score['x'][pp_score['ppscore'] != 0].to_list()
    #
    #     if self.verbosity > 0:
    #         print('[AutoML] Selected {} features with Predictive Power Score'.format(len(pp_cols)))
    #     return pp_cols

    def _sel_gini_impurity(self):
        """
        Calculates Feature Importance with Random Forest, aka Mean Decrease in Gini Impurity
        Symmetric correlation based on multiple features and multiple trees ensemble
        """
        if self.verbosity > 0:
            print('[AutoML] Analysing features with a Random Forest')

        # Set model
        rs = np.random.RandomState(seed=236868)
        if self.mode == 'regression':
            rf = RandomForestRegressor(random_state=rs).fit(self.x, self.y)
        elif self.mode == 'classification' or self.mode == 'multiclass':
            rf = RandomForestClassifier(random_state=rs).fit(self.x, self.y)
        else:
            raise ValueError('Method not implemented')

        # Get features
        fi = rf.feature_importances_
        sfi = fi.sum()
        ind = np.flip(np.argsort(fi))

        # Add to class attribute
        self.featureImportance['rf'] = (self.x.keys()[ind].to_list(), fi[ind].tolist())

        # Info Threshold
        ind_keep = [ind[i] for i in range(len(ind)) if fi[ind[:i]].sum() <= self.selectionCutoff * sfi]
        threshold = self.x.keys()[ind_keep].to_list()

        # Info increment
        ind_keep = [ind[i] for i in range(len(ind)) if fi[i] > sfi * self.selectionIncrement]
        increment = self.x.keys()[ind_keep].to_list()

        if self.verbosity > 0:
            print(f'[AutoML] Selected {len(threshold)} features with {self.selectionCutoff * 100}% RF threshold')
            print(f'[AutoML] Selected {len(increment)} features with {self.selectionIncrement * 100}% RF increment')
        return threshold, increment

    def _sel_shap(self):
        """
        Calculates Shapely values, which can be used as a measure of feature importance.
        """
        if self.verbosity > 0:
            print('[AutoML] Analysing features with Shapely Additive Explanations')
        # Get base model
        base = None
        if self.mode == 'regression':
            base = CatBoostRegressor()
        elif self.mode == 'classification' or self.mode == 'multiclass':
            base = CatBoostClassifier()

        # Fit
        base.fit(self.x, self.y)

        # Get SHAP values
        explainer = TreeExplainer(base.model)
        values = np.mean(np.abs(explainer.shap_values(self.x, self.y)), axis=0)
        values_sum = np.sum(values)
        ind = np.flip(np.argsort(values))

        # Add to class attribute
        self.featureImportance['shap'] = (self.x.keys()[ind].to_list(), values[ind].tolist())

        # Threshold
        ind_keep = [ind[i] for i in range(len(ind)) if values[ind[:i]].sum() <= self.selectionCutoff * values_sum]
        threshold = self.x.keys()[ind_keep].to_list()

        # Increment
        ind_keep = [ind[i] for i in range(len(ind)) if values[ind[i]] > values_sum * self.selectionIncrement * 10]
        increment = self.x.keys()[ind_keep].to_list()

        if self.verbosity > 0:
            print(f'[AutoML] Selected {len(threshold)} features with {self.selectionCutoff * 100}% Shap threshold')
            print(f"[AutoML] Selected {len(increment)} features with {self.selectionIncrement * 1000}% Shap increment")

        return threshold, increment

    # Deprecated
    # def _borutapy(self):
    #     if self.verbosity > 0:
    #         print('[AutoML] Analysing features with Boruta')
    #     rf = None
    #     if self.mode == 'regression':
    #         rf = RandomForestRegressor()
    #     elif self.mode == 'classification' or self.mode == 'multiclass':
    #         rf = RandomForestClassifier()
    #     selector = BorutaPy(rf, n_estimators='auto', verbose=0)
    #     selector.fit(self.x.to_numpy(), self.y.to_numpy())
    #     bp_cols = self.x.keys()[selector.support_].to_list()
    #     if self.verbosity > 0:
    #         print('[AutoML] Selected {} features with Boruta'.format(len(bp_cols)))
    #     return bp_cols

    def get_required_features(self, features: list):
        """
        Generate a list of required input features from a list of extracted features.
        """
        # Identify type of features and set class variables
        self.linearFeatures = [k for k in features if '__sub__' in k or '__add__' in k]
        self.crossFeatures = [k for k in features if '__x__' in k or '__d__' in k]
        self.trigonometricFeatures = [k for k in features if 'sin__' in k or 'cos__' in k]
        self.inverseFeatures = [k for k in features if 'inv__' in k]
        self.kMeansFeatures = [k for k in features if 'dist__' in k]
        self.diffFeatures = [k for k in features if '__diff__' in k]
        self.laggedFeatures = [k for k in features if '__lag__' in k]
        self.datetimeFeatures = [k for k in features if '__dt__' in k]
        original_features = [k for k in features if '__' not in k]

        # Fill missing features for normalization
        required = copy.copy(original_features)
        required += list(itertools.chain.from_iterable([s.split('__')[::2] for s in self.linearFeatures]))
        required += list(itertools.chain.from_iterable([s.split('__')[::2] for s in self.crossFeatures]))
        required += [s.split('__')[1] for s in self.trigonometricFeatures]
        required += [s[5:] for s in self.inverseFeatures]
        required += [s.split('__diff__')[0] for s in self.diffFeatures]
        required += [s.split('__lag__')[0] for s in self.laggedFeatures]
        required += [s.split('__dt__')[0] for s in self.datetimeFeatures]
        if len(self.kMeansFeatures) != 0:
            required += list(self._centers.keys())

        # Remove duplicates from required
        return list(set(required))
