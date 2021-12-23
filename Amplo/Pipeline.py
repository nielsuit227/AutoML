import re
import os
import time
import copy
import json
import Amplo
import joblib
import shutil
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from shap import TreeExplainer

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from Amplo import Utils

from Amplo.AutoML.Sequencer import Sequencer
from Amplo.AutoML.Modeller import Modeller
from Amplo.AutoML.DataSampler import DataSampler
from Amplo.AutoML.DataExplorer import DataExplorer
from Amplo.AutoML.DataProcesser import DataProcesser
from Amplo.AutoML.DriftDetector import DriftDetector
from Amplo.AutoML.FeatureProcesser import FeatureProcesser
from Amplo.Regressors.StackingRegressor import StackingRegressor
from Amplo.Classifiers.StackingClassifier import StackingClassifier

from .GridSearch.BaseGridSearch import BaseGridSearch
from .GridSearch.HalvingGridSearch import HalvingGridSearch
from .GridSearch.OptunaGridSearch import OptunaGridSearch

from .Documenting.MultiDocumenting import MultiDocumenting
from .Documenting.BinaryDocumenting import BinaryDocumenting
from .Documenting.RegressionDocumenting import RegressionDocumenting


class Pipeline:

    def __init__(self, **kwargs):
        """
        Automated Machine Learning Pipeline for tabular data.
        Designed for predictive maintenance applications, failure identification, failure prediction, condition
        monitoring, etc.

        Parameters
        ----------
        Main Parameters:
        target [str]: Column name of the output/dependent/regressand variable.
        name [str]: Name of the project (for documentation)
        version [int]: Pipeline version (set automatically)
        mode [str]: 'classification' or 'regression'
        objective [str]: from sklearn metrics and scoring

        Data Processor:
        int_cols [list[str]]: Column names of integer columns
        float_cols [list[str]]: Column names of float columns
        date_cols [list[str]]: Column names of datetime columns
        cat_cols [list[str]]: Column names of categorical columns
        missing_values [str]: [DataProcessing] - 'remove', 'interpolate', 'mean' or 'zero'
        outlier_removal [str]: [DataProcessing] - 'clip', 'boxplot', 'z-score' or 'none'
        z_score_threshold [int]: [DataProcessing] If outlier_removal = 'z-score', the threshold is adaptable
        include_output [bool]: Whether to include output in the training data (sensible only with sequencing)

        Feature Processor:
        extract_features [bool]: Whether or not to use FeatureProcessing module
        information_threshold : [FeatureProcessing] Threshold for removing co-linear features
        feature_timeout [int]: [FeatureProcessing] Time budget for feature processing
        max_lags [int]: [FeatureProcessing] Maximum lags for lagged features to analyse
        max_diff [int]: [FeatureProcessing] Maximum differencing order for differencing features

        Sequencing:
        sequence [bool]: [Sequencing] Whether or not to use Sequence module
        seq_back [int or list[int]]: Input time indices
            If list -> includes all integers within the list
            If int -> includes that many samples back
        seq_forward [int or list[int]: Output time indices
            If list -> includes all integers within the list.
            If int -> includes that many samples forward.
        seq_shift [int]: Shift input / output samples in time
        seq_diff [int]:  Difference the input & output, 'none', 'diff' or 'log_diff'
        seq_flat [bool]: Whether to return a matrix (True) or Tensor (Flat)

        Modelling:
        standardize [bool]: Whether to standardize input/output data
        shuffle [bool]: Whether to shuffle the samples during cross-validation
        cv_splits [int]: How many cross-validation splits to make
        store_models [bool]: Whether to store all trained model files

        Grid Search:
        grid_search_type [str]: Which method to use 'optuna', 'halving', 'base'
        grid_search_time_budget : Time budget for grid search
        grid_search_candidates : Parameter evaluation budget for grid search
        grid_search_iterations : Model evaluation budget for grid search

        Stacking:
        stacking [bool]: Whether to create a stacking model at the end

        Production:
        preprocess_function [str]: Add custom code for the prediction function, useful for production. Will be executed
            with exec, can be multiline. Uses data as input.

        Flags:
        plot_eda [bool]: Whether or not to run Exploratory Data Analysis
        process_data [bool]: Whether or not to force data processing
        document_results [bool]: Whether or not to force documenting
        no_dirs [bool]: Whether to create files or not
        verbose [int]: Level of verbosity
        """
        # Production initiation
        self.bestModel = None
        self.settings = None
        self.mainDir = 'AutoML/'

        # Copy arguments
        ##################
        # Main Settings
        self.target = re.sub('[^a-z0-9]', '_', kwargs.get('target', '').lower())
        self.name = kwargs.get('name', 'AutoML')
        self.version = kwargs.get('version', None)
        self.mode = kwargs.get('mode', None)
        self.objective = kwargs.get('objective', None)

        # Data Processor
        self.intCols = kwargs.get('int_cols', [])
        self.floatCols = kwargs.get('float_cols', [])
        self.dateCols = kwargs.get('date_cols', [])
        self.catCols = kwargs.get('cat_cols', [])
        self.missingValues = kwargs.get('missing_values', 'zero')
        self.outlierRemoval = kwargs.get('outlier_removal', 'clip')
        self.zScoreThreshold = kwargs.get('z_score_threshold', 4)
        self.includeOutput = kwargs.get('include_output', False)

        # Feature Processor
        self.extractFeatures = kwargs.get('extract_features', True)
        self.informationThreshold = kwargs.get('information_threshold', 0.999)
        self.featureTimeout = kwargs.get('feature_timeout', 3600)
        self.maxLags = kwargs.get('max_lags', 0)
        self.maxDiff = kwargs.get('max_diff', 0)

        # Sequencer
        self.sequence = kwargs.get('sequence', False)
        self.sequenceBack = kwargs.get('seq_back', 1)
        self.sequenceForward = kwargs.get('seq_forward', 1)
        self.sequenceShift = kwargs.get('seq_shift', 0)
        self.sequenceDiff = kwargs.get('seq_diff', 'none')
        self.sequenceFlat = kwargs.get('seq_flat', True)

        # Modelling
        self.standardize = kwargs.get('standardize', False)
        self.shuffle = kwargs.get('shuffle', True)
        self.cvSplits = kwargs.get('cv_shuffle', 10)
        self.storeModels = kwargs.get('store_models', False)

        # Grid Search Parameters
        self.gridSearchType = kwargs.get('grid_search_type', 'optuna')
        self.gridSearchTimeout = kwargs.get('grid_search_time_budget', 3600)
        self.gridSearchCandidates = kwargs.get('grid_search_candidates', 250)
        self.gridSearchIterations = kwargs.get('grid_search_iterations', 3)

        # Stacking
        self.stacking = kwargs.get('stacking', False)

        # Production
        self.preprocessFunction = kwargs.get('preprocess_function', None)

        # Flags
        self.plotEDA = kwargs.get('plot_eda', False)
        self.processData = kwargs.get('process_data', True)
        self.documentResults = kwargs.get('document_results', True)
        self.verbose = kwargs.get('verbose', 0)
        self.noDirs = kwargs.get('no_dirs', False)

        # Monitoring
        self._prediction_time = None
        self._main_predictors = None

        # Checks
        assert self.mode in [None, 'regression', 'classification'], 'Supported modes: regression, classification.'
        assert 0 < self.informationThreshold < 1, 'Information threshold needs to be within [0, 1]'
        assert self.maxLags < 50, 'Max_lags too big. Max 50.'
        assert self.maxDiff < 5, 'Max diff too big. Max 5.'
        assert self.gridSearchType.lower() in ['base', 'halving', 'optuna'], 'Grid Search Type must be Base, Halving ' \
                                                                             'or Optuna'

        # Advices
        if self.includeOutput and not self.sequence:
            warnings.warn('[AutoML] IMPORTANT: strongly advices to not include output without sequencing.')

        # Objective & Scorer
        self.scorer = None
        if self.objective is not None:
            assert isinstance(self.objective, str), 'Objective needs to be a string'
            assert self.objective in metrics.SCORERS.keys(), 'Metric not supported, look at sklearn.metrics'
            self.scorer = metrics.SCORERS[self.objective]

        # Instance initiating
        self.data = None
        self.x = None
        self.y = None
        self.featureSets = None
        self.results = None
        self.n_classes = None
        self.is_fitted = False

        # Required sub-classes
        self.dataSampler = DataSampler()
        self.dataProcesser = DataProcesser()
        self.dataSequencer = Sequencer()
        self.featureProcesser = FeatureProcesser()
        self.driftDetector = DriftDetector()

        # Create dirs
        if not self.noDirs:
            self._create_dirs()
            self._load_version()

        # Store Pipeline Settings
        self.settings = {'pipeline': kwargs, 'validation': {}, 'feature_set': ''}

    # User Pointing Functions
    def get_settings(self) -> dict:
        """
        Get settings to recreate fitted object.
        """
        assert self.is_fitted, "Pipeline not yet fitted."
        return self.settings

    def load_settings(self, settings: dict):
        """
        Restores a pipeline from settings.

        Parameters
        ----------
        settings [dict]: Pipeline settings
        """
        # Set parameters
        settings['pipeline']['no_dirs'] = True
        self.__init__(**settings['pipeline'])
        self.settings = settings
        self.dataProcesser.load_settings(settings['data_processing'])
        self.featureProcesser.load_settings(settings['feature_processing'])
        if 'drift_detector' in settings:
            self.driftDetector = DriftDetector(
                num_cols=self.dataProcesser.float_cols + self.dataProcesser.int_cols,
                cat_cols=self.dataProcesser.cat_cols,
                date_cols=self.dataProcesser.date_cols
            ).load_weights(settings['drift_detector'])

    def load_model(self, model: object):
        """
        Restores a trained model
        """
        assert type(model).__name__ == self.settings['model']
        self.bestModel = model
        self.is_fitted = True

    def fit(self, *args, **kwargs):
        """
        Fit the full autoML pipeline.

        1. Data Processing
        Cleans all the data. See @DataProcessing
        2. (optional) Exploratory Data Analysis
        Creates a ton of plots which are helpful to improve predictions manually
        3. Feature Processing
        Extracts & Selects. See @FeatureProcessing
        4. Initial Modelling
        Runs various off the shelf models with default parameters for all feature sets
        If Sequencing is enabled, this is where it happens, as here, the feature set is generated.
        5. Grid Search
        Optimizes the hyper parameters of the best performing models
        6. (optional) Create Stacking model
        7. (optional) Create documentation
        8. Prepare Production Files
        Nicely organises all required scripts / files to make a prediction

        Parameters
        ----------
        data [pd.DataFrame] - Single dataframe with input and output data.
        """
        # Starting
        print('\n\n*** Starting Amplo AutoML - {} ***\n\n'.format(self.name))

        # Reading data
        data = self._read_data(*args, **kwargs)

        # Detect mode (classification / regression)
        self._mode_detector(data)

        # Preprocess Data
        self._data_processing(data)

        # Run Exploratory Data Analysis
        self._eda()

        # Balance data
        self._data_sampling()

        # Sequence
        self._sequencing()

        # Extract and select features
        self._feature_processing()

        # Standardize
        # Standardizing assures equal scales, equal gradients and no clipping.
        # Therefore it needs to be after sequencing & feature processing, as this alters scales
        self._standardizing()

        # Run initial models
        self._initial_modelling()

        # Optimize Hyper parameters
        self.grid_search()

        # Create stacking model
        self._create_stacking()

        # Prepare production files
        self._prepare_production_files()

        self.is_fitted = True
        print('[AutoML] All done :)')

    def convert_data(self, x: pd.DataFrame, preprocess: bool = True) -> [pd.DataFrame, pd.Series]:
        """
        Function that uses the same process as the pipeline to clean data.
        Useful if pipeline is pickled for production

        Parameters
        ----------
        data [pd.DataFrame]: Input features
        """
        # Convert to Pandas
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=[f"Feature_{i}" for i in range(x.shape[1])])

        # Custom code
        if self.preprocessFunction is not None and preprocess:
            ex_globals = {'data': x}
            exec(self.preprocessFunction, ex_globals)
            x = ex_globals['data']

        # Process data
        x = self.dataProcesser.transform(x)
        if x.astype('float32').replace([np.inf, -np.inf], np.nan).isna().sum().sum() != 0:
            raise ValueError(f"Column(s) with NaN: {list(x.keys()[x.isna().sum() > 0])}")

        # Drift Check
        self.driftDetector.check(x)

        # Split output
        y = None
        if self.target in x.keys():
            y = x[self.target]
            if not self.includeOutput:
                x = x.drop(self.target, axis=1)

        # Sequence
        if self.sequence:
            x, y = self.dataSequencer.convert(x, y)

        # Convert Features
        x = self.featureProcesser.transform(x, self.settings['feature_set'])

        # Standardize
        if self.standardize:
            x, y = self._transform_standardize(x, y)

        # Return
        return x, y

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Full script to make predictions. Uses 'Production' folder with defined or latest version.

        Parameters
        ----------
        data [pd.DataFrame]: data to do prediction on
        """
        start_time = time.time()
        assert self.is_fitted, "Pipeline not yet fitted."

        # Print
        if self.verbose > 0:
            print('[AutoML] Predicting with {}, v{}'.format(type(self.bestModel).__name__, self.version))

        # Convert
        x, y = self.convert_data(data)

        # Predict
        if self.mode == 'regression' and self.standardize:
            predictions = self._inverse_standardize(self.bestModel.predict(x))
        else:
            predictions = self.bestModel.predict(x)

        # Stop timer
        self._prediction_time = (time.time() - start_time) / len(x) * 1000

        # Calculate main predictors
        self._get_main_predictors(x)

        return predictions

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Returns probabilistic prediction, only for classification.

        Parameters
        ----------
        data [pd.DataFrame]: data to do prediction on
        """
        start_time = time.time()
        assert self.is_fitted, "Pipeline not yet fitted."
        assert self.mode == 'classification', 'Predict_proba only available for classification'
        assert hasattr(self.bestModel, 'predict_proba'), '{} has no attribute predict_proba'.format(
            type(self.bestModel).__name__)

        # Print
        if self.verbose > 0:
            print('[AutoML] Predicting with {}, v{}'.format(type(self.bestModel).__name__, self.version))

        # Convert data
        x, y = self.convert_data(data)

        # Predict
        prediction = self.bestModel.predict_proba(x)

        # Stop timer
        self._prediction_time = (time.time() - start_time) / len(x) * 1000

        # Calculate main predictors
        self._get_main_predictors(x)

        return prediction

    # Fit functions
    def _read_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        To support Pandas & Numpy, with just data and x, y, this function reads the data and loads into desired format.

        Parameters
        ----------
        args / kwargs:
            x (pd.DataFrame): input
            y (pd.Series): output
            data (pd.DataFrame): data
        """
        assert len(args) + len(kwargs) != 0, "No data provided."

        # Handle args
        if len(args) + len(kwargs) == 1:
            if len(args) == 1:
                data = args[0]
            elif len(kwargs) == 1:
                assert 'data' in kwargs, "'data' argument missing"
                data = kwargs['data']
            else:
                raise ValueError('No data provided')

            # Test data
            assert isinstance(data, pd.DataFrame), "With only 1 argument, data must be a Pandas Dataframe."
            assert self.target != '', 'No target string provided'
            assert self.target in Utils.clean_keys(data).keys(), 'Target column missing'

        elif len(args) + len(kwargs) == 2:
            if len(args) == 2:
                x, y = args
            elif len(kwargs) == 2:
                assert 'x' in kwargs and 'y' in kwargs, "'x' or 'y' argument missing"
                x, y = kwargs['x'], kwargs['y']
            else:
                raise ValueError('Cannot understand partially named arguments...')

            # Parse data
            assert isinstance(x, (np.ndarray, pd.Series, pd.DataFrame)), "Unsupported data type for 'x'"
            if isinstance(x, pd.Series):
                data = pd.DataFrame(x)
            elif isinstance(x, np.ndarray):
                data = pd.DataFrame(x, columns=[f"Feature_{i}" for i in range(x.shape[1])])
            else:
                data = x

            # Check (and update) target
            if self.target == '':
                self.target = 'target'

            # Add target
            data[self.target] = y

        else:
            raise ValueError('Incorrect number of arguments.')

        return data

    def _mode_detector(self, data: pd.DataFrame):
        """
        Detects the mode (Regression / Classification)
        :param data: Data to detect mode on
        """
        # Only run if mode is not provided
        if self.mode is None:

            # Classification if string
            if data[self.target].dtype == str or \
                    data[self.target].nunique() < 0.1 * len(data):
                self.mode = 'classification'
                self.objective = 'neg_log_loss'

            # Else regression
            else:
                self.mode = 'regression'
                self.objective = 'neg_mean_absolute_error'
            self.scorer = metrics.SCORERS[self.objective]

            # Copy to settings
            self.settings['pipeline']['mode'] = self.mode
            self.settings['pipeline']['objective'] = self.objective

            # Print
            if self.verbose > 0:
                print(f"[AutoML] Setting mode to {self.mode} & objective to {self.objective}.")
        return

    def _data_processing(self, data: pd.DataFrame):
        """
        Organises the data cleaning. Heavy lifting is done in self.dataProcesser, but settings etc. needs
        to be organised.
        """
        self.dataProcesser = DataProcesser(target=self.target, int_cols=self.intCols, float_cols=self.floatCols,
                                           date_cols=self.dateCols, cat_cols=self.catCols,
                                           missing_values=self.missingValues,
                                           outlier_removal=self.outlierRemoval, z_score_threshold=self.zScoreThreshold)

        try:
            # Load data
            data = pd.read_csv(self.mainDir + 'Data/Cleaned_v{}.csv'.format(self.version), index_col='index')

            # Load settings
            self.settings['data_processing'] = json.load(open(self.mainDir + 'Settings/Cleaning_v{}.json'
                                                              .format(self.version), 'r'))
            self.dataProcesser.load_settings(self.settings['data_processing'])

            if self.verbose > 0:
                print('[AutoML] Loaded Cleaned Data')

        except FileNotFoundError:
            # Cleaning
            data = self.dataProcesser.fit_transform(data)

            # Store data
            data.to_csv(self.mainDir + 'Data/Cleaned_v{}.csv'.format(self.version), index_label='index')

            # Save settings
            self.settings['data_processing'] = self.dataProcesser.get_settings()
            json.dump(self.settings['data_processing'], open(self.mainDir + 'Settings/Cleaning_v{}.json'
                                                             .format(self.version), 'w'))

        # If no columns were provided, load them from data processor
        if self.dateCols is None:
            self.dateCols = self.settings['data_processing']['date_cols']
        if self.intCols is None:
            self.dateCols = self.settings['data_processing']['int_cols']
        if self.floatCols is None:
            self.floatCols = self.settings['data_processing']['float_cols']
        if self.catCols is None:
            self.catCols = self.settings['data_processing']['cat_cols']

        # Split and store in memory
        self.data = data
        self.y = data[self.target]
        self.x = data
        if self.includeOutput is False:
            self.x = self.x.drop(self.target, axis=1)

        # Assert classes in case of classification
        self.n_classes = self.y.nunique()
        if self.mode == 'classification':
            if self.n_classes >= 50:
                warnings.warn('More than 20 classes, you may want to reconsider classification mode')
            if set(self.y) != set([i for i in range(len(set(self.y)))]):
                raise ValueError('Classes should be [0, 1, ...]')

    def _eda(self):
        if self.plotEDA:
            print('[AutoML] Starting Exploratory Data Analysis')
            eda = DataExplorer(self.x, y=self.y,
                               mode=self.mode,
                               folder=self.mainDir,
                               version=self.version)
            eda.run()

    def _data_sampling(self):
        """
        Only run for classification problems. Balances the data using imblearn.
        Does not guarantee to return balanced classes. (Methods are data dependent)
        """
        # Only necessary for classification
        self.dataSampler = DataSampler(method='both', margin=0.1, cv_splits=self.cvSplits, shuffle=self.shuffle,
                                       fast_run=False, objective=self.objective)
        if self.mode == 'classification':
            # Check if exists
            try:
                # Load
                data = pd.read_csv(self.mainDir + 'Data/Balanced_v{}.csv'.format(self.version), index_col='index')

                # Split
                self.y = data[self.target]
                self.x = data
                if self.includeOutput is False:
                    self.x = self.x.drop(self.target, axis=1)

                if self.verbose > 0:
                    print('[AutoML] Loaded Balanced data')

            except FileNotFoundError:
                # Fit & Resample
                self.x, self.y = self.dataSampler.fit_resample(self.x, self.y)

                # Store
                data = copy.copy(self.x)
                data[self.target] = self.y
                data.to_csv(self.mainDir + 'Data/Balanced_v{}.csv'.format(self.version), index_label='index')

    def _sequencing(self):
        """
        Sequences the data. Useful mostly for problems where older samples play a role in future values.
        The settings of this module are NOT AUTOMATIC
        """
        self.dataSequencer = Sequencer(back=self.sequenceBack, forward=self.sequenceForward,
                                       shift=self.sequenceShift, diff=self.sequenceDiff)
        if self.sequence:
            try:
                # Load data
                data = pd.read_csv(self.mainDir + 'Data/Sequence_v{}.csv'.format(self.version), index_col='index')

                # Split and set to memory
                self.y = data[self.target]
                self.x = data
                if not self.includeOutput:
                    self.x = self.x.drop(self.target, axis=1)

                if self.verbose > 0:
                    print('[AutoML] Loaded Extracted Features')

            except FileNotFoundError:
                print('[AutoML] Sequencing data')
                self.x, self.y = self.dataSequencer.convert(self.x, self.y)

                # Save
                data = copy.deepcopy(self.x)
                data[self.target] = copy.deepcopy(self.y)
                data.to_csv(self.mainDir + 'Data/Sequence_v{}.csv'.format(self.version), index_label='index')

    def _feature_processing(self):
        """
        Organises feature processing. Heavy lifting is done in self.featureProcesser, but settings, etc.
        needs to be organised.
        """
        self.featureProcesser = FeatureProcesser(mode=self.mode, max_lags=self.maxLags, max_diff=self.maxDiff,
                                                 extract_features=self.extractFeatures, timeout=self.featureTimeout,
                                                 information_threshold=self.informationThreshold)
        # Check if exists
        try:
            # Loading data
            self.x = pd.read_csv(self.mainDir + 'Data/Extracted_v{}.csv'.format(self.version), index_col='index')

            # Loading settings
            self.settings['feature_processing'] = json.load(open(self.mainDir + 'Settings/Features_v{}.json'
                                                                 .format(self.version), 'r'))
            self.featureProcesser.load_settings(self.settings['feature_processing'])
            self.featureSets = self.settings['feature_processing']['featureSets']

            if self.verbose > 0:
                print('[AutoML] Loaded Extracted Features')
        except FileNotFoundError:
            print('[AutoML] Starting Feature Processor')

            # Transform data
            self.x, self.featureSets = self.featureProcesser.fit_transform(self.x, self.y)

            # Store data
            self.x.to_csv(self.mainDir + 'Data/Extracted_v{}.csv'.format(self.version), index_label='index')

            # Save settings
            self.settings['feature_processing'] = self.featureProcesser.get_settings()
            json.dump(self.settings['feature_processing'], open(self.mainDir + 'Settings/Features_v{}.json'
                                                                .format(self.version), 'w'))

    def _standardizing(self):
        """
        Wrapper function to determine whether to fit or load
        """
        # Return if standardize is off
        if not self.standardize:
            return

        # Load if exists
        try:
            self.settings['standardize'] = json.load(open(self.mainDir + 'Settings/Standardize_v{}.json'
                                                          .format(self.version), 'r'))

        # Otherwise fits
        except FileNotFoundError:
            self._fit_standardize(self.x, self.y)

            # Store Settings
            json.dump(self.settings['standardize'], open(self.mainDir + 'Settings/Standardize_v{}.json'
                                                         .format(self.version), 'w'))

        # And transform
        self.x, self.y = self._transform_standardize(self.x, self.y)

    def _initial_modelling(self):
        """
        Runs various models to see which work well.
        """
        # Load existing results
        if 'Results.csv' in os.listdir(self.mainDir):

            # Load results
            self.results = pd.read_csv(self.mainDir + 'Results.csv')

            # Printing here as we load it
            results = self.results[np.logical_and(
                self.results['version'] == self.version,
                self.results['type'] == 'Initial modelling'
            )]
            for fs in set(results['dataset']):
                print('[AutoML] Initial Modelling for {} ({})'.format(fs, len(self.featureSets[fs])))
                fsr = results[results['dataset'] == fs]
                for i in range(len(fsr)):
                    row = fsr.iloc[i]
                    print('[AutoML] {} {}: {:.4f} \u00B1 {:.4f}'.format(row['model'].ljust(40), self.objective,
                                                                        row['mean_objective'], row['std_objective']))

        # Check if this version has been modelled
        if self.results is None or self.version not in self.results['version'].values:

            # Iterate through feature sets
            for feature_set, cols in self.featureSets.items():

                # Skip empty sets
                if len(cols) == 0:
                    print('[AutoML] Skipping {} features, empty set'.format(feature_set))
                    continue
                print('[AutoML] Initial Modelling for {} features ({})'.format(feature_set, len(cols)))

                # Do the modelling
                modeller = Modeller(mode=self.mode, shuffle=self.shuffle, store_models=self.storeModels,
                                    objective=self.objective, dataset=feature_set,
                                    store_results=False, folder=self.mainDir + 'Models/')
                results = modeller.fit(self.x[cols], self.y)

                # Add results to memory
                results['type'] = 'Initial modelling'
                results['version'] = self.version
                if self.results is None:
                    self.results = results
                else:
                    self.results = self.results.append(results)

            # Save results
            self.results.to_csv(self.mainDir + 'Results.csv', index=False)

    def grid_search(self, model=None, feature_set: str = None, parameter_set: str = None):
        """
        Runs a grid search. By default, takes the self.results, and runs for the top 3 optimizations.
        There is the option to provide a model & feature_set, but both have to be provided. In this case,
        the model & data set combination will be optimized.
        Implemented types, Base, Halving, Optuna

        Parameters
        ----------
        model [Object or str]- (optional) Which model to run grid search for.
        feature_set [str]- (optional) Which feature set to run grid search for 'rft', 'rfi' or 'pps'
        parameter_set [dict]- (optional) Parameter grid to optimize over
        """
        assert model is not None and feature_set is not None or model == feature_set, \
            'Model & feature_set need to be either both None or both provided.'
        # If arguments are provided
        if model is not None:

            # Get model string
            if isinstance(model, str):
                model = Utils.getModel(model, mode=self.mode, samples=len(self.x))

            # Organise existing results
            results = self.results[np.logical_and(
                self.results['model'] == type(model).__name__,
                self.results['version'] == self.version,
            )]
            results = self._sort_results(results[results['dataset'] == feature_set])

            # Check if exists and load
            if ('Hyper Parameter' == results['type']).any():
                print('[AutoML] Loading optimization results.')
                hyper_opt_results = results[results['type'] == 'Hyper Parameter']
                params = Utils.parse_json(hyper_opt_results.iloc[0]['params'])

            # Or run
            else:
                # Run grid search
                grid_search_results = self._sort_results(self._grid_search_iteration(model, parameter_set, feature_set))

                # Store results
                grid_search_results['model'] = type(model).__name__
                grid_search_results['version'] = self.version
                grid_search_results['dataset'] = feature_set
                grid_search_results['type'] = 'Hyper Parameter'
                self.results = self.results.append(grid_search_results)
                self.results.to_csv(self.mainDir + 'Results.csv', index=False)

                # Get params for validation
                params = Utils.parse_json(grid_search_results.iloc[0]['params'])

            # Validate
            if self.documentResults:
                self.document(model.set_params(**params), feature_set)
            return

        # If arguments aren't provided, run through promising models
        results = self._sort_results(self.results[np.logical_and(
            self.results['type'] == 'Initial modelling',
            self.results['version'] == self.version,
        )])
        for iteration in range(self.gridSearchIterations):
            # Grab settings
            settings = results.iloc[iteration] # IndexError
            model = Utils.getModel(settings['model'], mode=self.mode, samples=len(self.x))
            feature_set = settings['dataset']

            # Check whether exists
            model_results = self.results[np.logical_and(
                self.results['model'] == type(model).__name__,
                self.results['version'] == self.version,
            )]
            model_results = self._sort_results(model_results[model_results['dataset'] == feature_set])

            # If exists
            if ('Hyper Parameter' == model_results['type']).any():
                hyper_opt_res = model_results[model_results['type'] == 'Hyper Parameter']
                params = Utils.parse_json(hyper_opt_res.iloc[0]['params'])

            # Else run
            else:
                # For one model
                grid_search_results = self._sort_results(self._grid_search_iteration(
                    copy.deepcopy(model), parameter_set, feature_set))

                # Store
                grid_search_results['version'] = self.version
                grid_search_results['dataset'] = feature_set
                grid_search_results['type'] = 'Hyper Parameter'
                self.results = self.results.append(grid_search_results)
                self.results.to_csv(self.mainDir + 'Results.csv', index=False)
                params = Utils.parse_json(grid_search_results.iloc[0]['params'])

            # Validate
            if self.documentResults:
                self.document(model.set_params(**params), feature_set)

    def _create_stacking(self):
        """
        Based on the best performing models, in addition to cheap models based on very different assumptions,
        A stacking model is optimized to enhance/combine the performance of the models.
        --> should contain a large variety of models
        --> classifiers need predict_proba
        --> level 1 needs to be ordinary least squares
        """
        if self.stacking:
            print('[AutoML] Creating Stacking Ensemble')

            # Select feature set that has been picked most often for hyper parameter optimization
            results = self._sort_results(self.results[np.logical_and(
                self.results['type'] == 'Hyper Parameter',
                self.results['version'] == self.version,
            )])
            feature_set = results['dataset'].value_counts().index[0]
            results = results[results['dataset'] == feature_set]
            print('[AutoML] Selected Stacking feature set: {}'.format(feature_set))

            # Create Stacking Model Params
            n_stacking_models = 3
            stacking_models_str = results['model'].unique()[:n_stacking_models]
            stacking_models_params = [Utils.parse_json(results.iloc[np.where(results['model'] == sms)[0][0]]['params'])
                                      for sms in stacking_models_str]
            stacking_models = dict([(sms, stacking_models_params[i]) for i, sms in enumerate(stacking_models_str)])
            print('[AutoML] Stacked models: {}'.format(list(stacking_models.keys())))

            # Add samples & Features
            stacking_models['n_samples'], stacking_models['n_features'] = self.x.shape

            # Prepare Stack
            if self.mode == 'regression':
                stack = StackingRegressor(**stacking_models)
                cv = KFold(n_splits=self.cvSplits, shuffle=self.shuffle)

            elif self.mode == 'classification':
                stack = StackingClassifier(**stacking_models)
                cv = StratifiedKFold(n_splits=self.cvSplits, shuffle=self.shuffle)
            else:
                raise NotImplementedError('Unknown mode')

            # Cross Validate
            x, y = self.x[self.featureSets[feature_set]].to_numpy(), self.y.to_numpy()
            score = []
            times = []
            for (t, v) in tqdm(cv.split(x, y)):
                start_time = time.time()
                xt, xv, yt, yv = x[t], x[v], y[t].reshape((-1)), y[v].reshape((-1))
                model = copy.deepcopy(stack)
                model.fit(xt, yt)
                score.append(self.scorer(model, xv, yv))
                times.append(time.time() - start_time)

            # Output Results
            print('[AutoML] Stacking result:')
            print('[AutoML] {}:        {:.2f} \u00B1 {:.2f}'.format(self.objective, np.mean(score), np.std(score)))
            self.results = self.results.append({
                'date': datetime.today().strftime('%d %b %y'),
                'model': type(stack).__name__,
                'dataset': feature_set,
                'params': json.dumps(stack.get_params()),
                'mean_objective': np.mean(score),
                'std_objective': np.std(score),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'version': self.version,
                'type': 'Stacking',
            }, ignore_index=True)
            self.results.to_csv(self.mainDir + 'Results.csv', index=False)

            # Document
            if self.documentResults:
                self.document(stack, feature_set)

    def document(self, model, feature_set: str):
        """
        Loads the model and features and initiates the outside Documenting class.

        Parameters
        ----------
        model [Object or str]- (optional) Which model to run grid search for.
        feature_set [str]- (optional) Which feature set to run grid search for 'rft', 'rfi' or 'pps'
        """
        # Get model
        if isinstance(model, str):
            model = Utils.getModel(model, mode=self.mode, samples=len(self.x))

        # Checks
        assert feature_set in self.featureSets.keys(), 'Feature Set not available.'
        if os.path.exists(self.mainDir + 'Documentation/v{}/{}_{}.pdf'.format(
                self.version, type(model).__name__, feature_set)):
            print('[AutoML] Documentation existing for {} v{} - {} '.format(
                type(model).__name__, self.version, feature_set))
            return
        if len(model.get_params()) == 0:
            warnings.warn('[Documenting] Supplied model has no parameters!')

        # Run validation
        print('[AutoML] Creating Documentation for {} - {}'.format(type(model).__name__, feature_set))
        if self.mode == 'classification' and self.n_classes == 2:
            documenting = BinaryDocumenting(self)
        elif self.mode == 'classification':
            documenting = MultiDocumenting(self)
        elif self.mode == 'regression':
            documenting = RegressionDocumenting(self)
        else:
            raise ValueError('Unknown mode.')
        documenting.create(model, feature_set)

        # Append to settings
        self.settings['validation']['{}_{}'.format(type(model).__name__, feature_set)] = documenting.outputMetrics

    def _prepare_production_files(self, model=None, feature_set: str = None, params: dict = None):
        """
        Prepares files necessary to deploy a specific model / feature set combination.
        - Model.joblib
        - Settings.json
        - Report.pdf

        Parameters
        ----------
        model [string] : (optional) Model file for which to prep production files
        feature_set [string] : (optional) Feature set for which to prep production files
        params [optional, dict]: (optional) Model parameters for which to prep production files, if None, takes best.
        """
        # Path variable
        prod_path = self.mainDir + 'Production/v{}/'.format(self.version)

        # Create production folder
        if not os.path.exists(prod_path):
            os.mkdir(prod_path)

        # Filter results for this version
        results = self._sort_results(self.results[self.results['version'] == self.version])

        # Filter results if model is provided
        if model is not None:
            # Take name if model instance is given
            if not isinstance(model, str):
                model = type(model).__name__

            # Filter results
            results = self._sort_results(results[results['model'] == model])

        # Filter results if feature set is provided
        if feature_set is not None:
            results = self._sort_results(results[results['dataset'] == feature_set])

        # Get best parameters
        if params is None:
            params = results.iloc[0]['params']

        # Otherwise Find best
        model = results.iloc[0]['model']
        feature_set = results.iloc[0]['dataset']
        params = Utils.parse_json(params)

        # Printing action
        if self.verbose > 0:
            print('[AutoML] Preparing Production files for {}, {}, {}'.format(model, feature_set, params))

        # Try to load model
        if os.path.exists(prod_path + 'Model.joblib'):
            self.bestModel = joblib.load(prod_path + 'Model.joblib')

        # Rerun if not existent, or different than desired
        if not os.path.exists(prod_path + 'Model.joblib') or \
            type(self.bestModel).__name__ != model or \
                self.bestModel.get_params() != params:

            # Stacking needs to be created by a separate script :/
            if 'Stacking' in model:
                if self.mode == 'regression':
                    self.bestModel = StackingRegressor(n_samples=len(self.x), n_features=len(self.x.keys()))
                elif self.mode == 'classification':
                    self.bestModel = StackingClassifier(n_samples=len(self.x), n_features=len(self.x.keys()))
                else:
                    raise NotImplementedError("Mode not set")

            else:
                # Model
                self.bestModel = Utils.getModel(model, mode=self.mode, samples=len(self.x))

            # Set params, train, & save
            self.bestModel.set_params(**params)
            self.bestModel.fit(self.x[self.featureSets[feature_set]], self.y)
            joblib.dump(self.bestModel, self.mainDir + 'Production/v{}/Model.joblib'.format(self.version))

            if self.verbose > 0:
                print('[AutoML] Model fully fitted, in-sample {}: {:4f}'
                      .format(self.objective, self.scorer(self.bestModel, self.x[self.featureSets[feature_set]],
                                                          self.y)))

        else:
            if self.verbose > 0:
                print('[AutoML] Loading existing model file.')

        # Update pipeline settings
        self.settings['pipeline']['verbose'] = 0
        self.settings['model'] = model  # The string
        self.settings['params'] = params
        self.settings['feature_set'] = feature_set
        self.settings['features'] = self.featureSets[feature_set]
        self.settings['amplo_version'] = Amplo.__version__ if hasattr(Amplo, '__version__') else 'dev'

        # Prune Data Processor
        required_features = self.featureProcesser.get_required_features(self.featureSets[feature_set])
        self.dataProcesser.prune_features(required_features)
        self.settings['data_processing'] = self.dataProcesser.get_settings()

        # Fit Drift Detector
        self.driftDetector = DriftDetector(
            num_cols=self.dataProcesser.float_cols + self.dataProcesser.int_cols,
            cat_cols=self.dataProcesser.cat_cols,
            date_cols=self.dataProcesser.date_cols
        )
        self.driftDetector.fit(self.data)
        self.driftDetector.fit_output(self.bestModel, self.x[self.featureSets[feature_set]])
        self.settings['drift_detector'] = self.driftDetector.get_weights()

        # Report
        if not os.path.exists('{}Documentation/v{}/{}_{}.pdf'.format(self.mainDir, self.version, model, feature_set)):
            self.document(self.bestModel, feature_set)
        shutil.copy('{}Documentation/v{}/{}_{}.pdf'.format(self.mainDir, self.version, model, feature_set),
                    '{}Production/v{}/Report.pdf'.format(self.mainDir, self.version))

        # Save settings
        json.dump(self.settings, open(self.mainDir + 'Production/v{}/Settings.json'
                                      .format(self.version), 'w'), indent=4)

        return self

    # Support Functions
    def _load_version(self):
        """
        Upon start, loads version
        """
        # No need if version is set
        if self.version is not None:
            return

        # Read changelog (if existent)
        if os.path.exists(self.mainDir + 'changelog.txt'):
            with open(self.mainDir + 'changelog.txt', 'r') as f:
                changelog = f.read()
        else:
            changelog = ''

        # Find production folders
        completed_versions = len(os.listdir(self.mainDir + 'Production'))
        started_versions = len(changelog.split('\n')) - 1

        # For new runs
        if started_versions == 0:
            with open(self.mainDir + 'changelog.txt', 'w') as f:
                f.write('v1: Initial Run\n')
            self.version = 1

        # If last run was completed and we start a new
        elif started_versions == completed_versions and self.processData:
            self.version = started_versions + 1
            with open(self.mainDir + 'changelog.txt', 'a') as f:
                f.write('v{}: {}\n'.format(self.version, input('Changelog v{}:\n'.format(self.version))))

        # If no new run is started (either continue or rerun)
        else:
            self.version = started_versions

        if self.verbose > 0:
            print(f'[AutoML] Setting Version {self.version}')

    def _create_dirs(self):
        folders = ['', 'EDA', 'Data', 'Features', 'Documentation', 'Production', 'Settings']
        for folder in folders:
            if not os.path.exists(self.mainDir + folder):
                os.makedirs(self.mainDir + folder)

    def sort_results(self, results: pd.DataFrame) -> pd.DataFrame:
        return self._sort_results(results)

    def _fit_standardize(self, x: pd.DataFrame, y: pd.Series):
        """
        Fits a standardization parameters and returns the transformed data
        """
        # Check if existing
        if os.path.exists(self.mainDir + 'Settings/Standardize_{}.json'.format(self.version)):
            self.settings['standardize'] = json.load(open(self.mainDir + 'Settings/Standardize_{}.json'
                                                          .format(self.version), 'r'))
            return

        # Fit Input
        cat_cols = [k for lst in self.settings['data_processing']['dummies'].values() for k in lst]
        features = [k for k in x.keys() if k not in self.dateCols and k not in cat_cols]
        means_ = x[features].mean(axis=0)
        stds_ = x[features].std(axis=0)
        stds_[stds_ == 0] = 1
        settings = {
            'input': {
                'features': features,
                'means': means_.to_list(),
                'stds': stds_.to_list(),
            }
        }

        # Fit Output
        if self.mode == 'regression':
            std = y.std()
            settings['output'] = {
                'mean': y.mean(),
                'std': std if std != 0 else 1,
            }

        self.settings['standardize'] = settings

    def _transform_standardize(self, x: pd.DataFrame, y: pd.Series) -> [pd.DataFrame, pd.Series]:
        """
        Standardizes the input and output with values from settings.

        Parameters
        ----------
        x [pd.DataFrame]: Input data
        y [pd.Series]: Output data
        """
        # Input
        assert self.settings['standardize'], "Standardize settings not found."

        # Pull from settings
        features = self.settings['standardize']['input']['features']
        means = self.settings['standardize']['input']['means']
        stds = self.settings['standardize']['input']['stds']

        # Filter if not all features are present
        if len(x.keys()) < len(features):
            indices = [[i for i, j in enumerate(features) if j == k][0] for k in x.keys()]
            features = [features[i] for i in indices]
            means = [means[i] for i in indices]
            stds = [stds[i] for i in indices]

        # Transform Input
        x[features] = (x[features] - means) / stds

        # Transform output (only with regression)
        if self.mode == 'regression':
            y = (y - self.settings['standardize']['output']['mean']) / self.settings['standardize']['output']['std']

        return x, y

    def _inverse_standardize(self, y: pd.Series) -> pd.Series:
        """
        For predictions, transform them back to application scales.
        Parameters
        ----------
        y [pd.Series]: Standardized output

        Returns
        -------
        y [pd.Series]: Actual output
        """
        assert self.settings['standardize'], "Standardize settings not found"
        return y * self.settings['standardize']['output']['std'] + self.settings['standardize']['output']['mean']

    @staticmethod
    def _sort_results(results: pd.DataFrame) -> pd.DataFrame:
        return results.sort_values('worst_case', ascending=False)

    def _get_best_params(self, model, feature_set: str) -> dict:
        # Filter results for model and version
        results = self.results[np.logical_and(
            self.results['model'] == type(model).__name__,
            self.results['version'] == self.version,
        )]

        # Filter results for feature set & sort them
        results = self._sort_results(results[results['dataset'] == feature_set])

        # Warning for unoptimized results
        if 'Hyper Parameter' not in results['type'].values:
            warnings.warn('Hyper parameters not optimized for this combination')

        # Parse & return best parameters (regardless of if it's optimized)
        return Utils.parse_json(results.iloc[0]['params'])

    def _grid_search_iteration(self, model, parameter_set: str, feature_set: str):
        """
        INTERNAL | Grid search for defined model, parameter set and feature set.
        """
        print('\n[AutoML] Starting Hyper Parameter Optimization for {} on {} features ({} samples, {} features)'.format(
            type(model).__name__, feature_set, len(self.x), len(self.featureSets[feature_set])))

        # Cross-Validator
        cv = StratifiedKFold(n_splits=self.cvSplits, shuffle=self.shuffle)
        if self.mode == 'regression':
            cv = KFold(n_splits=self.cvSplits, shuffle=self.shuffle)

        # Select right hyper parameter optimizer
        if self.gridSearchType == 'base':
            grid_search = BaseGridSearch(model, params=parameter_set, cv=cv, scoring=self.objective,
                                         candidates=self.gridSearchCandidates, timeout=self.gridSearchTimeout,
                                         verbose=self.verbose)
        elif self.gridSearchType == 'halving':
            grid_search = HalvingGridSearch(model, params=parameter_set, cv=cv, scoring=self.objective,
                                            candidates=self.gridSearchCandidates, verbose=self.verbose)
        elif self.gridSearchType == 'optuna':
            grid_search = OptunaGridSearch(model, timeout=self.gridSearchTimeout, cv=cv,
                                           candidates=self.gridSearchCandidates, scoring=self.objective,
                                           verbose=self.verbose)
        else:
            raise NotImplementedError('Only Base, Halving and Optuna are implemented.')
        # Get results
        results = grid_search.fit(self.x[self.featureSets[feature_set]], self.y)
        return results.sort_values('worst_case', ascending=False)

    def _get_main_predictors(self, data):
        """
        Using Shapely Additive Explanations, this function calculates the main predictors for a given
        prediction and sets them into the class' memory.
        """
        if type(self.bestModel).__name__ in ['SVC', 'RidgeClassifier', 'LinearRegression']:
            return
        elif type(self.bestModel).__module__[:5] == 'Amplo':
            explainer = TreeExplainer(self.bestModel.model)
        else:
            explainer = TreeExplainer(self.bestModel)

        # Get values
        shap_values = np.array(explainer.shap_values(data))

        # Shape them (for multiclass it outputs ndim=3, for binary/regression ndim=2)
        if shap_values.ndim == 3:
            shap_values = shap_values[1]

        # Take mean over samples
        shap_values = np.mean(shap_values, axis=0)

        # Sort them
        inds = sorted(range(len(shap_values)), key=lambda x: -abs(shap_values[x]))

        # Set class attribute
        self._main_predictors = dict([(data.keys()[i], float(abs(shap_values[i]))) for i in inds])
