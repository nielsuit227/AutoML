import itertools
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
from typing import Union
from pathlib import Path
from datetime import datetime
from shap import TreeExplainer
from shap import KernelExplainer

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from Amplo import Utils
from Amplo.AutoML.Sequencer import Sequencer
from Amplo.AutoML.Modeller import Modeller
from Amplo.AutoML.DataSampler import DataSampler
from Amplo.AutoML.DataExplorer import DataExplorer
from Amplo.AutoML.DataProcessor import DataProcessor
from Amplo.AutoML.DriftDetector import DriftDetector
from Amplo.AutoML.FeatureProcessor import FeatureProcessor
from Amplo.AutoML.IntervalAnalyser import IntervalAnalyser
from Amplo.Classifiers.StackingClassifier import StackingClassifier
from Amplo.Documenting import BinaryDocumenting
from Amplo.Documenting import MultiDocumenting
from Amplo.Documenting import RegressionDocumenting
from Amplo.GridSearch import BaseGridSearch
from Amplo.GridSearch import HalvingGridSearch
from Amplo.GridSearch import OptunaGridSearch
from Amplo.Observation import DataObserver
from Amplo.Observation import ProductionObserver
from Amplo.Regressors.StackingRegressor import StackingRegressor


class Pipeline:

    def __init__(self, **kwargs):
        """
        Automated Machine Learning Pipeline for tabular data.
        Designed for predictive maintenance applications, failure identification, failure prediction, condition
        monitoring, etc.

        Parameters
        ----------
        Main Parameters:
        main_dir [str]: Main directory of Pipeline (for documentation)
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
        extract_features [bool]: Whether to use FeatureProcessing module
        information_threshold : [FeatureProcessing] Threshold for removing co-linear features
        feature_timeout [int]: [FeatureProcessing] Time budget for feature processing
        max_lags [int]: [FeatureProcessing] Maximum lags for lagged features to analyse
        max_diff [int]: [FeatureProcessing] Maximum differencing order for differencing features

        Interval Analyser:
        interval_analyse [bool]: Whether to use IntervalAnalyser module
            Note that this has no effect when data from ``self._read_data`` is not multi-indexed

        Sequencing:
        sequence [bool]: [Sequencing] Whether to use Sequence module
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
        grid_search_type [Optional[str]]: Which method to use 'optuna', 'halving', 'base' or None
        grid_search_time_budget : Time budget for grid search
        grid_search_candidates : Parameter evaluation budget for grid search
        grid_search_iterations : Model evaluation budget for grid search

        Stacking:
        stacking [bool]: Whether to create a stacking model at the end

        Production:
        preprocess_function [str]: Add custom code for the prediction function, useful for production. Will be executed
            with exec, can be multiline. Uses data as input.

        Flags:
        logging_level [Optional[Union[int, str]]]: Logging level for warnings, info, etc.
        plot_eda [bool]: Whether to run Exploratory Data Analysis
        process_data [bool]: Whether to force data processing
        document_results [bool]: Whether to force documenting
        no_dirs [bool]: Whether to create files or not
        verbose [int]: Level of verbosity
        """

        # Copy arguments
        ##################
        # Main Settings
        self.mainDir = kwargs.get('main_dir', 'AutoML/')
        self.target = re.sub('[^a-z0-9]', '_', kwargs.get('target', '').lower())
        self.name = kwargs.get('name', 'AutoML')
        self.version = kwargs.get('version', None)
        self.mode = kwargs.get('mode', None)
        self.objective = kwargs.get('objective', None)

        # Data Processor
        self.intCols = kwargs.get('int_cols', None)
        self.floatCols = kwargs.get('float_cols', None)
        self.dateCols = kwargs.get('date_cols', None)
        self.catCols = kwargs.get('cat_cols', None)
        self.missingValues = kwargs.get('missing_values', 'zero')
        self.outlierRemoval = kwargs.get('outlier_removal', 'clip')
        self.zScoreThreshold = kwargs.get('z_score_threshold', 4)
        self.includeOutput = kwargs.get('include_output', False)

        # Balancer
        self.balance = kwargs.get('balance', True)

        # Feature Processor
        self.extractFeatures = kwargs.get('extract_features', True)
        self.informationThreshold = kwargs.get('information_threshold', 0.999)
        self.featureTimeout = kwargs.get('feature_timeout', 3600)
        self.maxLags = kwargs.get('max_lags', 0)
        self.maxDiff = kwargs.get('max_diff', 0)

        # Interval Analyser
        self.useIntervalAnalyser = kwargs.get('interval_analyse', True)

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

        # Checks
        assert self.mode in [None, 'regression', 'classification'], 'Supported modes: regression, classification.'
        assert 0 < self.informationThreshold < 1, 'Information threshold needs to be within [0, 1]'
        assert self.maxLags < 50, 'Max_lags too big. Max 50.'
        assert self.maxDiff < 5, 'Max diff too big. Max 5.'
        assert self.gridSearchType is None \
            or self.gridSearchType.lower() in ['base', 'halving', 'optuna'], \
            'Grid Search Type must be Base, Halving, Optuna or None'

        # Advices
        if self.includeOutput and not self.sequence:
            warnings.warn('[AutoML] IMPORTANT: strongly advices to not include output without sequencing.')

        # Create dirs
        if not self.noDirs:
            self._create_dirs()
            self._load_version()

        # Store Pipeline Settings
        self.settings = {'pipeline': kwargs, 'validation': {}, 'feature_set': ''}

        # Objective & Scorer
        self.scorer = None
        if self.objective is not None:
            assert isinstance(self.objective, str), 'Objective needs to be a string'
            assert self.objective in metrics.SCORERS.keys(), 'Metric not supported, look at sklearn.metrics'

        # Required sub-classes
        self.dataSampler = DataSampler()
        self.dataProcessor = DataProcessor()
        self.dataSequencer = Sequencer()
        self.featureProcessor = FeatureProcessor()
        self.intervalAnalyser = IntervalAnalyser()
        self.driftDetector = DriftDetector()

        # Instance initiating
        self.bestModel = None
        self._data = None
        self.featureSets = None
        self.results = None
        self.n_classes = None
        self.is_fitted = False

        # Monitoring
        logging_level = kwargs.get('logging_level', 'INFO')
        logging_dir = Path(self.mainDir) / 'app_logs.log' if not self.noDirs else None
        self.logger = Utils.logging.get_logger('AutoML', logging_dir, logging_level, capture_warnings=True)
        self._prediction_time = None
        self._main_predictors = None

    # User Pointing Functions
    def get_settings(self, version: int = None) -> dict:
        """
        Get settings to recreate fitted object.

        Parameters
        ----------
        version : int, optional
            Production version, defaults to current version
        """
        if version is None or version == self.version:
            assert self.is_fitted, "Pipeline not yet fitted."
            return self.settings
        else:
            settings_path = self.mainDir + f'Production/v{self.version}/Settings.json'
            assert Path(settings_path).exists(), 'Cannot load settings from nonexistent version'
            return json.load(open(settings_path, 'r'))

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
        self.dataProcessor.load_settings(settings['data_processing'])
        self.featureProcessor.load_settings(settings['feature_processing'])
        # TODO: load_settings for IntervalAnalyser (not yet implemented)
        if 'drift_detector' in settings:
            self.driftDetector = DriftDetector(
                num_cols=self.dataProcessor.float_cols + self.dataProcessor.int_cols,
                cat_cols=self.dataProcessor.cat_cols,
                date_cols=self.dataProcessor.date_cols
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
        Fit the full AutoML pipeline.
            1. Prepare data for training
            2. Train / optimize models
            3. Prepare Production Files
                Nicely organises all required scripts / files to make a prediction

        Parameters
        ----------
        args
            For data reading - Propagated to `self.data_preparation`
        kwargs
            For data reading (propagated to `self.data_preparation`) AND
            for production filing (propagated to `self.conclude_fitting`)
        """
        # Starting
        print('\n\n*** Starting Amplo AutoML - {} ***\n\n'.format(self.name))

        # Prepare data for training
        self.data_preparation(*args, **kwargs)

        # Train / optimize models
        self.model_training(**kwargs)

        # Conclude fitting
        self.conclude_fitting(**kwargs)

    def data_preparation(self, *args, **kwargs):
        """
        Prepare data for modelling
            1. Data Processing
                Cleans all the data. See @DataProcessing
            2. (optional) Exploratory Data Analysis
                Creates a ton of plots which are helpful to improve predictions manually
            3. Feature Processing
                Extracts & Selects. See @FeatureProcessing

        Parameters
        ----------
        args
            For data reading - Propagated to `self._read_data`
        kwargs
            For data reading - Propagated to `self._read_data`
        """
        # Reading data
        self._read_data(*args, **kwargs)

        # Check data
        obs = DataObserver(pipeline=self)
        obs.observe()

        # Detect mode (classification / regression)
        self._mode_detector()

        # Preprocess Data
        self._data_processing()

        # Run Exploratory Data Analysis
        self._eda()

        # Balance data
        self._data_sampling()

        # Sequence
        self._sequencing()

        # Extract and select features
        self._feature_processing()

        # Interval-analyze data
        self._interval_analysis()

        # Standardize
        # Standardizing assures equal scales, equal gradients and no clipping.
        # Therefore, it needs to be after sequencing & feature processing, as this alters scales
        self._standardizing()

    def model_training(self, **kwargs):
        """Train models

        1. Initial Modelling
            Runs various off the shelf models with default parameters for all feature sets
            If Sequencing is enabled, this is where it happens, as here, the feature set is generated.
        2. Grid Search
            Optimizes the hyperparameters of the best performing models
        3. (optional) Create Stacking model
        4. (optional) Create documentation

        Parameters
        ----------
        kwargs : optional
            Keyword arguments that will be passed to `self.grid_search`.
        """
        # Run initial models
        self._initial_modelling()

        # Optimize Hyper parameters
        self.grid_search(**kwargs)

        # Create stacking model
        self._create_stacking()

    def conclude_fitting(self, *, model=None, feature_set=None, params=None, **kwargs):
        """
        Prepare production files that are necessary to deploy a specific
        model / feature set combination

        Creates or modifies the following files
            - ``Model.joblib`` (production model)
            - ``Settings.json`` (model settings)
            - ``Report.pdf`` (training report)

        Parameters
        ----------
        model : str or list of str, optional
            Model file for which to prepare production files. If multiple, selects the best.
        feature_set : str or list of str, optional
            Feature set for which to prepare production files. If multiple, selects the best.
        params : dict, optional
            Model parameters for which to prepare production files.
            Default: takes the best parameters
        kwargs
            Collecting container for keyword arguments that are passed through `self.fit()`.
        """
        # Set up production path
        prod_dir = self.mainDir + f'Production/v{self.version}/'
        Path(prod_dir).mkdir(exist_ok=True)

        # Parse arguments
        model, feature_set, params = self._parse_production_args(model, feature_set, params)

        # Verbose printing
        if self.verbose > 0:
            print(f'[AutoML] Preparing Production files for {model}, {feature_set}, {params}')

        # Set best model (`self.bestModel`)
        self._prepare_production_model(prod_dir + 'Model.joblib', model, feature_set, params)

        # Set and store production settings
        self._prepare_production_settings(prod_dir + 'Settings.json', model, feature_set, params)

        # Observe production
        # TODO[TS, 25.05.2022]: Currently, we are observing the data also here.
        #  However, in a future version we probably will only observe the data
        #  directly after :func:`_read_data()`. For now we wait...
        obs = ProductionObserver(pipeline=self)
        obs.observe()
        self.settings['production_observation'] = obs.observations

        # Report
        report_path = self.mainDir + f'Documentation/v{self.version}/{model}_{feature_set}.pdf'
        if not Path(report_path).exists():
            self.document(self.bestModel, feature_set)
        shutil.copy(report_path, prod_dir + 'Report.pdf')

        # Finish
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
        x = self.dataProcessor.transform(x)

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
        x = self.featureProcessor.transform(x, self.settings['feature_set'])

        # Standardize
        if self.standardize:
            x, y = self._transform_standardize(x, y)

        # NaN test -- datetime should be taken care of by now
        if x.astype('float32').replace([np.inf, -np.inf], np.nan).isna().sum().sum() != 0:
            raise ValueError(f"Column(s) with NaN: {list(x.keys()[x.isna().sum() > 0])}")

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
    def _read_data(self, x=None, y=None, *, data=None, **kwargs):
        """
        Reads and loads data into desired format.

        Expects to receive:
            1. Both, ``x`` and ``y`` (-> features and target), or
            2. Either ``x`` or ``data`` (-> dataframe or path to folder)

        Parameters
        ----------
        x : np.ndarray or pd.Series or pd.DataFrame or str or Path, optional
            x-data (input) OR acts as ``data`` parameter when param ``y`` is empty
        y : np.ndarray or pd.Series, optional
            y-data (target)
        data : pd.DataFrame or str or Path, optional
            Contains both, x and y, OR provides a path to folder structure
        kwargs
            Collecting container for keyword arguments that are passed through `self.fit()`.

        Returns
        -------
        Pipeline
        """
        assert x is not None or data is not None, 'No data provided'
        assert (x is not None) ^ (data is not None), 'Setting both, `x` and `data`, is ambiguous'

        # Labels are provided separately
        if y is not None:
            # Check data
            x = x if x is not None else data
            assert x is not None, 'Parameter ``x`` is not set'
            assert isinstance(x, (np.ndarray, pd.Series, pd.DataFrame)), 'Unsupported data type for parameter ``x``'
            assert isinstance(y, (np.ndarray, pd.Series)), 'Unsupported data type for parameter ``y``'

            # Set target manually if not defined
            if self.target == '':
                self.target = 'target'

            # Parse x-data
            if isinstance(x, np.ndarray):
                x = pd.DataFrame(x)
            elif isinstance(x, pd.Series):
                x = pd.DataFrame(x)
            # Parse y-data
            if isinstance(y, np.ndarray):
                y = pd.Series(y, index=x.index)
            y.name = self.target

            # Check data
            assert all(x.index == y.index), '``x`` and ``y`` indices do not match'
            if self.target in x.columns:
                assert all(x[self.target] == y), 'Target column co-exists in both, ``x`` and ``y`` data, ' \
                                                 f'but has not equal content. Rename the column ``{self.target}`` ' \
                                                 'in ``x`` or set a (different) target in initialization.'

            # Concatenate x and y
            data = pd.concat([x, y], axis=1)

        # Set data parameter in case it is provided through parameter ``x``
        data = data if data is not None else x
        metadata = None

        # A path was provided to read out (multi-indexed) data
        if isinstance(data, (str, Path)):
            # Set target manually if not defined
            if self.target == '':
                self.target = 'target'
            # Parse data
            data, metadata = Utils.io.merge_logs(data, self.target)

        # Test data
        assert self.target != '', 'No target string provided'
        assert self.target in data.columns, 'Target column missing'
        assert len(data.columns) == data.columns.nunique(), 'Columns have no unique names'

        # Parse data
        y = data[self.target]
        x = data.drop(self.target, axis=1)
        if isinstance(x.columns, pd.RangeIndex):
            x.columns = [f'Feature_{i}' for i in range(x.shape[1])]
        # Concatenate x and y
        data = pd.concat([x, y], axis=1)

        # Save data
        self.set_data(data)

        # Store metadata in settings
        self.settings['file_metadata'] = metadata or dict()

        return self

    def has_new_training_data(self):
        # Return True if no previous version exists
        if self.version == 1:
            return True

        # Get previous and current file metadata
        curr_metadata = self.settings['file_metadata']
        last_metadata = self.get_settings(self.version - 1)['file_metadata']

        # Check each settings file
        for file_id in curr_metadata:
            # Get file specific metadata
            curr = curr_metadata[file_id]
            last = last_metadata.get(file_id, dict())
            # Compare metadata
            same_folder = curr['folder'] == last.get('folder')
            same_file = curr['file'] == last.get('file')
            same_mtime = curr['last_modified'] == last.get('last_modified')
            if not all([same_folder, same_file, same_mtime]):
                return False

        return True

    def _mode_detector(self):
        """
        Detects the mode (Regression / Classification)
        """
        # Only run if mode is not provided
        if self.mode is None:

            # Classification if string
            if self.y.dtype == str or self.y.nunique() < 0.1 * len(self.data):
                self.mode = 'classification'
                self.objective = self.objective or 'neg_log_loss'

            # Else regression
            else:
                self.mode = 'regression'
                self.objective = self.objective or 'neg_mean_absolute_error'

            # Set scorer
            self.scorer = metrics.SCORERS[self.objective]

            # Copy to settings
            self.settings['pipeline']['mode'] = self.mode
            self.settings['pipeline']['objective'] = self.objective

            # Print
            if self.verbose > 0:
                print(f"[AutoML] Setting mode to {self.mode} & objective to {self.objective}.")
        return

    def _data_processing(self):
        """
        Organises the data cleaning. Heavy lifting is done in self.dataProcessor, but settings etc. needs
        to be organised.
        """
        self.dataProcessor = DataProcessor(target=self.target, int_cols=self.intCols, float_cols=self.floatCols,
                                           date_cols=self.dateCols, cat_cols=self.catCols,
                                           missing_values=self.missingValues,
                                           outlier_removal=self.outlierRemoval, z_score_threshold=self.zScoreThreshold)

        # Set paths
        data_path = self.mainDir + f'Data/Cleaned_v{self.version}.csv'
        settings_path = self.mainDir + f'Settings/Cleaning_v{self.version}.json'

        if Path(data_path).exists() and Path(settings_path).exists():
            # Load data
            data = self._read_csv(data_path)
            self.set_data(data)

            # Load settings
            self.settings['data_processing'] = json.load(open(settings_path, 'r'))
            self.dataProcessor.load_settings(self.settings['data_processing'])

            if self.verbose > 0:
                print('[AutoML] Loaded Cleaned Data')

        else:
            # Cleaning
            data = self.dataProcessor.fit_transform(self.data)
            self.set_data(data)

            # Store data
            self._write_csv(self.data, data_path)

            # Save settings
            self.settings['data_processing'] = self.dataProcessor.get_settings()
            json.dump(self.settings['data_processing'], open(settings_path, 'w'))

        # If no columns were provided, load them from data processor
        if self.dateCols is None:
            self.dateCols = self.settings['data_processing']['date_cols']
        if self.intCols is None:
            self.dateCols = self.settings['data_processing']['int_cols']
        if self.floatCols is None:
            self.floatCols = self.settings['data_processing']['float_cols']
        if self.catCols is None:
            self.catCols = self.settings['data_processing']['cat_cols']

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
        self.dataSampler = DataSampler(method='both', margin=0.1, cv_splits=self.cvSplits, shuffle=self.shuffle,
                                       fast_run=False, objective=self.objective)

        # Set paths
        data_path = self.mainDir + f'Data/Balanced_v{self.version}.csv'

        # Only necessary for classification
        if self.mode == 'classification' and self.balance:

            if Path(data_path).exists():
                # Load data
                data = self._read_csv(data_path)
                self.set_data(data)

                if self.verbose > 0:
                    print('[AutoML] Loaded Balanced data')

            else:
                # Fit and resample
                print('[AutoML] Resampling data')
                x, y = self.dataSampler.fit_resample(self.x, self.y)

                # Store
                self._set_xy(x, y)
                self._write_csv(self.data, data_path)

    def _sequencing(self):
        """
        Sequences the data. Useful mostly for problems where older samples play a role in future values.
        The settings of this module are NOT AUTOMATIC
        """
        self.dataSequencer = Sequencer(back=self.sequenceBack, forward=self.sequenceForward,
                                       shift=self.sequenceShift, diff=self.sequenceDiff)

        # Set paths
        data_path = self.mainDir + f'Data/Sequence_v{self.version}.csv'

        if self.sequence:

            if Path(data_path).exists():
                # Load data
                data = self._read_csv(data_path)
                self.set_data(data)

                if self.verbose > 0:
                    print('[AutoML] Loaded Extracted Features')

            else:
                # Sequencing
                print('[AutoML] Sequencing data')
                x, y = self.dataSequencer.convert(self.x, self.y)

                # Store
                self._set_xy(x, y)
                self._write_csv(self.data, data_path)

    def _feature_processing(self):
        """
        Organises feature processing. Heavy lifting is done in self.featureProcessor, but settings, etc.
        needs to be organised.
        """
        self.featureProcessor = FeatureProcessor(mode=self.mode, max_lags=self.maxLags, max_diff=self.maxDiff,
                                                 extract_features=self.extractFeatures, timeout=self.featureTimeout,
                                                 information_threshold=self.informationThreshold)

        # Set paths
        data_path = self.mainDir + f'Data/Extracted_v{self.version}.csv'
        settings_path = self.mainDir + f'Settings/Extracting_v{self.version}.json'

        if Path(data_path).exists() and Path(settings_path).exists():
            # Loading data
            x = self._read_csv(data_path)
            self._set_x(x)

            # Loading settings
            self.settings['feature_processing'] = json.load(open(settings_path, 'r'))
            self.featureProcessor.load_settings(self.settings['feature_processing'])
            self.featureSets = self.settings['feature_processing']['featureSets']

            if self.verbose > 0:
                print('[AutoML] Loaded Extracted Features')

        else:
            print('[AutoML] Starting Feature Processor')

            # Transform data
            x, self.featureSets = self.featureProcessor.fit_transform(self.x, self.y)

            # Store data
            self._set_x(x)
            self._write_csv(self.x, data_path)

            # Save settings
            self.settings['feature_processing'] = self.featureProcessor.get_settings()
            json.dump(self.settings['feature_processing'], open(settings_path, 'w'))

    def _interval_analysis(self):
        """
        Interval-analyzes the data using ``Amplo.AutoML.IntervalAnalyser``
        or resorts to pre-computed data, if present.
        """
        # Skip analysis when analysis is not possible and/or not desired
        is_interval_analyzable = len(self.x.index.names) == 2
        if not (self.useIntervalAnalyser and is_interval_analyzable):
            return

        self.intervalAnalyser = IntervalAnalyser(target=self.target)

        # Set paths
        data_path = self.mainDir + f'Data/Interval_Analyzed_v{self.version}.csv'
        settings_path = self.mainDir + f'Settings/Interval_Analysis_v{self.version}.json'

        if Path(data_path).exists():  # TODO: and Path(settings_path).exists():
            # Load data
            data = self._read_csv(data_path)
            self.set_data(data)

            # TODO implement `IntervalAnalyser.load_settings` and add to `self.load_settings`
            # # Load settings
            # self.settings['interval_analysis'] = json.load(open(settings_path, 'r'))
            # self.intervalAnalyser.load_settings(self.settings['interval_analysis'])

            if self.verbose > 0:
                print('[AutoML] Loaded interval-analyzed data')

        else:
            print(f'[AutoML] Interval-analyzing data')

            # Transform data
            data = self.intervalAnalyser.fit_transform(self.x, self.y)

            # Store data
            self.set_data(data)
            self._write_csv(self.data, data_path)

            # TODO implement `IntervalAnalyser.get_settings` and add to `self.get_settings`
            # # Save settings
            # self.settings['interval_analysis'] = self.intervalAnalyser.get_settings()
            # json.dump(self.settings['interval_analysis'], open(settings_path, 'w'))

    def _standardizing(self):
        """
        Wrapper function to determine whether to fit or load
        """
        # Return if standardize is off
        if not self.standardize:
            return

        # Set paths
        settings_path = self.mainDir + f'Settings/Standardize_v{self.version}.json'

        if Path(settings_path).exists():
            # Load data
            self.settings['standardize'] = json.load(open(settings_path, 'r'))

        else:
            # Fit data
            self._fit_standardize(self.x, self.y)

            # Store Settings
            json.dump(self.settings['standardize'], open(settings_path, 'w'))

        # Transform data
        x, y = self._transform_standardize(self.x, self.y)
        self._set_xy(x, y)

    def _initial_modelling(self):
        """
        Runs various models to see which work well.
        """

        # Set paths
        results_path = Path(self.mainDir) / 'Results.csv'

        # Load existing results
        if results_path.exists():

            # Load results
            self.results = pd.read_csv(results_path)

            # Printing here as we load it
            results = self.results[np.logical_and(
                self.results['version'] == self.version,
                self.results['type'] == 'Initial modelling'
            )]
            for fs in set(results['dataset']):
                print(f'[AutoML] Initial Modelling for {fs} ({len(self.featureSets[fs])})')
                fsr = results[results['dataset'] == fs]
                for i in range(len(fsr)):
                    row = fsr.iloc[i]
                    print(f'[AutoML] {row["model"].ljust(40)} {self.objective}: '
                          f'{row["mean_objective"]:.4f} \u00B1 {row["std_objective"]:.4f}')

        # Check if this version has been modelled
        if self.results is None or self.version not in self.results['version'].values:

            # Iterate through feature sets
            for feature_set, cols in self.featureSets.items():

                # Skip empty sets
                if len(cols) == 0:
                    print(f'[AutoML] Skipping {feature_set} features, empty set')
                    continue
                print(f'[AutoML] Initial Modelling for {feature_set} features ({len(cols)})')

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
                    self.results = pd.concat([self.results, results])

            # Save results
            self.results.to_csv(results_path, index=False)

    def grid_search(self, model=None, feature_set=None, parameter_set=None, **kwargs):
        """Runs a grid search.

        By default, takes ``self.results`` and runs for the top ``n=self.gridSearchIterations`` optimizations.
        There is the option to provide ``model`` and ``feature_set``, but **both** have to be provided. In this
        case, the model and dataset combination will be optimized.

        Implemented types, Base, Halving, Optuna.

        Parameters
        ----------
        model : list of (str or object) or object or str, optional
            Which model to run grid search for.
        feature_set : list of str or str, optional
            Which feature set to run gid search for. Must be provided when `model` is not None.
            Options: ``RFT``, ``RFI``, ``ShapThreshold`` or ``ShapIncrement``
        parameter_set : dict, optional
            Parameter grid to optimize over.

        Notes
        -----
        When both parameters, ``model`` and ``feature_set``, are provided, the grid search behaves as follows:
            - When both parameters are either of dtype ``str`` or have the same length, then grid search will
              treat them as pairs.
            - When one parameter is an iterable and the other parameter is either a string or an iterable
              of different length, then grid search will happen for each unique combination of these parameters.
        """

        # Skip grid search and set best initial model as best grid search parameters
        if self.gridSearchType is None or self.gridSearchIterations == 0:
            best_initial_model = self._sort_results(self.results[self.results['version'] == self.version]).iloc[:1]
            best_initial_model['type'] = 'Hyper Parameter'
            self.results = pd.concat([self.results, best_initial_model], ignore_index=True)
            return self

        # Define models
        if model is None:
            # Run through first best initial models (n=`self.gridSearchIterations`)
            selected_results = self.sort_results(self.results[np.logical_and(
                self.results['type'] == 'Initial modelling',
                self.results['version'] == self.version,
            )]).iloc[:self.gridSearchIterations]
            models = [Utils.utils.get_model(model_name, mode=self.mode, samples=len(self.x))
                      for model_name in selected_results['model']]
            feature_sets = selected_results['dataset']

        elif feature_set is None:
            raise AttributeError('When `model` is provided, `feature_set` cannot be None. '
                                 'Provide either both params or neither of them.')

        else:
            models = [Utils.utils.get_model(model, mode=self.mode, samples=len(self.x))] \
                if isinstance(model, str) else [model]
            feature_sets = [feature_set] if isinstance(feature_set, str) else list(feature_set)
            if len(models) != len(feature_sets):
                # Create each combination
                combinations = list(itertools.product(np.unique(models), np.unique(feature_sets)))
                models = [elem[0] for elem in combinations]
                feature_sets = [elem[1] for elem in combinations]

        # Iterate and grid search over each pair of model and feature_set
        for model, feature_set in zip(models, feature_sets):

            # Organise existing model results
            m_results = self.results[np.logical_and(
                self.results['model'] == type(model).__name__,
                self.results['version'] == self.version,
            )]
            m_results = self._sort_results(m_results[m_results['dataset'] == feature_set])

            # Skip grid search if optimized model already exists
            if ('Hyper Parameter' == m_results['type']).any():
                print('[AutoML] Loading optimization results.')
                grid_search_results = m_results[m_results['type'] == 'Hyper Parameter']

            # Run grid search otherwise
            else:
                # Run grid search for model
                grid_search_results = self._grid_search_iteration(model, parameter_set, feature_set)
                grid_search_results = self.sort_results(grid_search_results)

                # Store results
                grid_search_results['version'] = self.version
                grid_search_results['dataset'] = feature_set
                grid_search_results['type'] = 'Hyper Parameter'
                self.results = pd.concat([self.results, grid_search_results], ignore_index=True)
                self.results.to_csv(self.mainDir + 'Results.csv', index=False)

            # Validate
            if self.documentResults:
                params = Utils.io.parse_json(grid_search_results.iloc[0]['params'])
                # TODO: What about other than our custom models? They don't have `set_params()` method
                self.document(model.set_params(**params), feature_set)

        return self

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
            stacking_models_params = [Utils.io.parse_json(results.iloc[np.where(results['model'] == sms)[0][0]]['params'])
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
            model = Utils.utils.get_model(model, mode=self.mode, samples=len(self.x))

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

    def _parse_production_args(self, model=None, feature_set=None, params=None):
        """
        Parse production arguments. Selects the best model, feature set and parameter combination.

        Parameters
        ----------
        model : str or list of str, optional
            Model constraint(s)
        feature_set : str or list of str, optional
            Feature set constraint(s)
        params : dict, optional
            Parameter constraint(s)
        Returns
        -------
        model : str
            Best model given the `model` restraint(s).
        feature_set : str
            Best feature set given the `feature_set` restraint(s).
        params : dict
            Best model parameters given the `params` restraint(s).
        """
        if model is not None and not isinstance(model, str):
            # TODO: This issue is linked with AML-103 (in Jira)
            #  1. Add to method docstring that it accepts a model instance, too
            #  2. Change `if`-case to a single `isinstance(model, BasePredictor)`
            # Get model name
            model = type(model).__name__

        # Get results of current version
        results = self._sort_results(self.results[self.results['version'] == self.version])

        if model is not None:
            if isinstance(model, str):
                model = [model]
            # Filter results
            results = self._sort_results(results[results['model'].isin(model)])

        if feature_set is not None:
            if isinstance(feature_set, str):
                feature_set = [feature_set]
            # Filter results
            results = self._sort_results(results[results['dataset'].isin(feature_set)])

        if params is None:
            # Get best parameters
            params = results.iloc[0]['params']

        # Find the best allowed arguments
        model = results.iloc[0]['model']
        feature_set = results.iloc[0]['dataset']
        params = Utils.io.parse_json(results.iloc[0]['params'])

        return model, feature_set, params

    def _prepare_production_model(self, model_path, model, feature_set, params):
        """
        Prepare and store `self.bestModel` for production

        Parameters
        ----------
        model_path : str or Path
            Where to store model for production
        model : str, optional
            Model file for which to prepare production files
        feature_set : str, optional
            Feature set for which to prepare production files
        params : dict, optional
            Model parameters for which to prepare production files.
            Default: takes best parameters

        Returns
        -------
        model : str
            Updated name of model
        feature_set : str
            Updated name of feature set
        """

        model_path = Path(model_path)

        # Try to load model from file
        if model_path.exists():
            # Load model
            self.bestModel = joblib.load(Path(model_path))
            # Reset if it's not the desired model
            if type(self.bestModel).__name__ != model or self.bestModel.get_params() != params:
                self.bestModel = None
        else:
            self.bestModel = None

        # Set best model
        if self.bestModel is not None:
            if self.verbose > 0:
                print('[AutoML] Loading existing model file')

        else:
            # Make model
            if 'Stacking' in model:
                # Create stacking
                if self.mode == 'regression':
                    self.bestModel = StackingRegressor(n_samples=len(self.x), n_features=len(self.x.keys()))
                elif self.mode == 'classification':
                    self.bestModel = StackingClassifier(n_samples=len(self.x), n_features=len(self.x.keys()))
                else:
                    raise NotImplementedError("Mode not set")
            else:
                # Take model as is
                self.bestModel = Utils.utils.get_model(model, mode=self.mode, samples=len(self.x))

            # Set params, train
            self.bestModel.set_params(**params)
            self.bestModel.fit(self.x[self.featureSets[feature_set]], self.y)

            # Save model
            joblib.dump(self.bestModel, model_path)

            if self.verbose > 0:
                score = self.scorer(self.bestModel, self.x[self.featureSets[feature_set]], self.y)
                print(f'[AutoML] Model fully fitted, in-sample {self.objective}: {score:4f}')

        return model, feature_set

    def _prepare_production_settings(self, settings_path, model=None, feature_set=None, params=None):
        """
        Prepare `self.settings` for production and dump to file

        Parameters
        ----------
        settings_path : str or Path
            Where to save settings for production
        model : str, optional
            Model file for which to prepare production files
        feature_set : str, optional
            Feature set for which to prepare production files
        params : dict, optional
            Model parameters for which to prepare production files.
            Default: takes best parameters
        """
        assert self.bestModel is not None, '`self.bestModel` is not yet prepared'
        settings_path = Path(settings_path)

        # Update pipeline settings
        self.settings['version'] = self.version
        self.settings['pipeline']['verbose'] = self.verbose
        self.settings['model'] = model
        self.settings['params'] = params
        self.settings['feature_set'] = feature_set
        self.settings['features'] = self.featureSets[feature_set]
        self.settings['amplo_version'] = Amplo.__version__ if hasattr(Amplo, '__version__') else 'dev'

        # Prune Data Processor
        required_features = self.featureProcessor.get_required_features(self.featureSets[feature_set])
        self.dataProcessor.prune_features(required_features)
        self.settings['data_processing'] = self.dataProcessor.get_settings()

        # Fit Drift Detector
        self.driftDetector = DriftDetector(
            num_cols=self.dataProcessor.float_cols + self.dataProcessor.int_cols,
            cat_cols=self.dataProcessor.cat_cols,
            date_cols=self.dataProcessor.date_cols
        )
        self.driftDetector.fit(self.x)
        self.driftDetector.fit_output(self.bestModel, self.x[self.featureSets[feature_set]])
        self.settings['drift_detector'] = self.driftDetector.get_weights()

        # Save settings
        json.dump(self.settings, open(settings_path, 'w'), indent=4)

    # Getter Functions / Properties
    @property
    def data(self) -> Union[None, pd.DataFrame]:
        return self._data

    @property
    def x(self) -> pd.DataFrame:
        if self.data is None:
            raise AttributeError('Data is None')
        if self.includeOutput:
            return self.data
        return self.data.drop(self.target, axis=1)

    @property
    def y(self):
        if self.data is None:
            raise AssertionError('`self.data` is empty. Set a value with `set_data`')
        return self.data[self.target]

    @property
    def y_orig(self):
        enc_labels = self.y
        dec_labels = self.dataProcessor.decode_labels(enc_labels, except_not_fitted=False)
        return pd.Series(dec_labels, name=self.target, index=enc_labels.index)

    # Setter Functions
    def set_data(self, new_data: pd.DataFrame):
        assert isinstance(new_data, pd.DataFrame), 'Invalid data type'
        assert self.target in new_data, 'No target column present'
        assert len(new_data.columns) > 1, 'No feature column present'
        self._data = new_data

    def _set_xy(self, new_x: Union[np.ndarray, pd.DataFrame], new_y: Union[np.ndarray, pd.Series]):
        if not isinstance(new_y, pd.Series):
            new_y = pd.Series(new_y, name=self.target)
        new_data = pd.concat([new_x, new_y], axis=1)
        self.set_data(new_data)

    def _set_x(self, new_x: Union[np.ndarray, pd.DataFrame]):
        old_x = self.data.drop(self.target, axis=1)
        # Convert to dataframe
        if isinstance(new_x, np.ndarray) and new_x.shape == old_x.shape:
            new_x = pd.DataFrame(new_x, index=old_x.index, columns=old_x.columns)
        elif isinstance(new_x, np.ndarray):
            warnings.warn('Old x-data has more/less columns than new x-data. Setting dummy feature names...')
            new_x = pd.DataFrame(new_x, index=old_x.index, columns=[f'Feature_{i}' for i in range(new_x.shape[1])])
        # Assert that target is not in x-data
        if self.target in new_x:
            raise AttributeError('Target column name should not be in x-data')
        self._data = pd.concat([new_x, self.y], axis=1)

    def set_y(self, new_y: Union[np.ndarray, pd.Series]):
        self._data[self.target] = new_y

    # Support Functions
    @staticmethod
    def _read_csv(data_path) -> pd.DataFrame:
        """
        Read data from given path and set index or multi-index

        Parameters
        ----------
        data_path : str or Path
        """
        assert Path(data_path).suffix == '.csv', 'Expected a *.csv path'

        data = pd.read_csv(data_path)

        if {'index', 'log'}.issubset(data.columns):
            # Multi-index: case when IntervalAnalyser was used
            index = ['log', 'index']
        elif 'index' in data.columns:
            index = ['index']
        else:
            raise IndexError('No known index was found. '
                             'Expected to find at least a column named `index`.')
        return data.set_index(index)

    @staticmethod
    def _write_csv(data, data_path):
        """
        Write data to given path and set index if needed.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
        data_path : str or Path
        """
        assert Path(data_path).suffix == '.csv', 'Expected a *.csv path'

        # Set single-index if not already present
        if len(data.index.names) == 1 and data.index.name is None:
            data.index.name = 'index'
        # Raise error if unnamed index is present
        if None in data.index.names:
            raise IndexError(f'Found an unnamed index column ({list(data.index.names)}).')

        # Write data
        data.to_csv(data_path)

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
                notice = input(f'Changelog v{self.version}:\n')
                f.write(f'v{self.version}: {notice}\n')

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
        return Utils.io.parse_json(results.iloc[0]['params'])

    def _grid_search_iteration(self, model, parameter_set: Union[str, None], feature_set: str):
        """
        INTERNAL | Grid search for defined model, parameter set and feature set.
        """
        print(f'\n[AutoML] Starting Hyper Parameter Optimization for {type(model).__name__} on '
              f'{feature_set} features ({len(self.x)} samples, {len(self.featureSets[feature_set])} features)')

        # Cross-Validator
        cv_args = {'n_splits': self.cvSplits, 'shuffle': self.shuffle,
                   'random_state': 83847939 if self.shuffle else None}
        cv = KFold(**cv_args) if self.mode == 'regression' else StratifiedKFold(**cv_args)

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
        results = results.sort_values('worst_case', ascending=False)

        # Warn when best hyperparameters are close to predefined grid
        edge_params = grid_search.get_parameter_min_max()
        best_params = pd.Series(results['params'].iloc[0], name='best')
        params = edge_params.join(best_params, how='inner')

        def warn_when_too_close_to_edge(param: pd.Series, tol=0.01):
            # Min-max scaling
            scaled = np.array(param['best']) / (param['max'] - param['min'])
            # Check if too close and warn if so
            if not (tol < scaled < 1 - tol):
                warnings.warn(f'Optimal value for parameter is very close to edge case: '
                              f'{param.name}={param["best"]} (range: {param["min"]}...{param["max"]})')

        params.apply(lambda p: warn_when_too_close_to_edge(p), axis=1)

        return results

    def _get_main_predictors(self, data):
        """
        Using Shapely Additive Explanations, this function calculates the main predictors for a given
        prediction and sets them into the class' memory.
        """
        # Shap is not implemented for all models.
        if type(self.bestModel).__name__ in ['SVC', 'BaggingClassifier', 'RidgeClassifier', 'LinearRegression', 'SVR',
                                             'BaggingRegressor']:
            features = self.settings['feature_processing']['featureImportance']['shap'][0]
            values = self.settings['feature_processing']['featureImportance']['shap'][1]
            self._main_predictors = {features[i]: values[i] for i in range(len(features))}

        else:
            if type(self.bestModel).__module__[:5] == 'Amplo':
                shap_values = np.array(TreeExplainer(self.bestModel.model).shap_values(data))
            else:
                shap_values = np.array(TreeExplainer(self.bestModel).shap_values(data))

            # Shape them (for multiclass it outputs ndim=3, for binary/regression ndim=2)
            if shap_values.ndim == 3:
                shap_values = shap_values[1]

            # Take mean over samples
            shap_values = np.mean(shap_values, axis=0)

            # Sort them
            inds = sorted(range(len(shap_values)), key=lambda x: -abs(shap_values[x]))

            # Set class attribute
            self._main_predictors = dict([(data.keys()[i], float(abs(shap_values[i]))) for i in inds])

        return self._main_predictors
