#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import copy
import itertools
import json
import os
import time
import warnings
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Tuple, Union

import joblib
import numpy as np
import pandas as pd
from shap import TreeExplainer
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

import amplo
from amplo import utils
from amplo.automl.data_exploring import DataExplorer
from amplo.automl.data_processing import DataProcessor
from amplo.automl.data_sampling import DataSampler
from amplo.automl.drift_detection import DriftDetector
from amplo.automl.feature_processing import FeatureProcessor
from amplo.automl.interval_analysis import IntervalAnalyser
from amplo.automl.modelling import Modeller
from amplo.automl.sequencing import Sequencer
from amplo.base.objects import LoggingMixin
from amplo.classification.stacking import StackingClassifier
from amplo.grid_search import ExhaustiveGridSearch, HalvingGridSearch, OptunaGridSearch
from amplo.observation import DataObserver, ModelObserver
from amplo.regression.stacking import StackingRegressor
from amplo.validation import ModelValidator

__all__ = ["Pipeline"]


class Pipeline(LoggingMixin):
    """
    Automated Machine Learning Pipeline for tabular data.

    The pipeline is designed for predictive maintenance application, failure
    identification, failure prediction, condition monitoring, and more.

    Parameters
    ----------
    # Main parameters
    main_dir : str, default: "Auto_ML/"
        Main directory of pipeline (for documentation).
    target : str, optional
        Column name of the output variable.
    name : str, default: "AutoML"
        Name of the project (for documentation).
    version : int, default: 1
        Pipeline version. Will automatically increment when a version exists.
    mode : {None, "classification", "regression"}, default: None
        Pipeline mode.
    objective : str, optional
        Objective for training.
        Default for classification: "neg_log_loss".
        Default for regression: "mean_square_error".
    verbose : int, default: 1
        Verbosity of logging.

    # Data processing
    int_cols : list of str, optional
        Column names of integer columns (if None will detect automatically).
    float_cols : list of str, optional
        Column names of float columns (if None will detect automatically).
    date_cols : list of str, optional
        Column names of datetime column (if None will detect automatically).
    cat_cols : list of str, optional
        Column names of categorical columns (if None will detect automatically).
    missing_values : {"remove", "interpolate", "mean", "zero"}, default: "zero"
        How to treat missing values.
    outlier_removal : {"clip", "boxplot", "z-score", "none"}, default: "clip"
        How to treat outliers.
    z_score_threshold : int, default: 4
        When ``outlier_removal`` is "z-score", the threshold is adaptable.
    include_output : bool, default: False
        Whether to include output in the training data (sensible only with sequencing).

    # Balancing
    balance : bool, default: False
        Whether to balance data.

    # Feature processing
    extract_features : bool, default: True
        Whether to use the FeatureProcessing module to extract features.
    information_threshold : float, default: 0.999
        Threshold for removing collinear features.
    feature_timeout : int, default: 3600
        Time budget for feature processing.

    # Interval analysis
    interval_analyse : bool, default: False
        Whether to use the IntervalAnalyser module.

    # Sequencing
    sequence : bool, default: False
        Whether to use the Sequencer module.
    seq_back : int or list of int, default: 1
        Input time indices.
        If int: includes that many samples backward.
        If list of int: includes all integers within the list.
    seq_forward : int or list of int, default: 1
        Output time indices.
        If int: includes that many samples forward.
        If list of int: includes all integers within the list.
    seq_shift : int, default: 0
        Shift input / output samples in time.
    seq_diff : {"none", "diff", "log_diff"}, default: "none"
        Difference the input and output.
    seq_flat : bool, default: True
        Whether to return a matrix (True) or a tensor (False).

    # Modelling
    standardize : bool, default: False
        Whether to standardize the input/output data.
    cv_shuffle : bool, default: True
        Whether to shuffle the samples during cross-validation.
    cv_splits : int, default: 10
        How many cross-validation splits to make.
    store_models : bool, default: False
        Whether to store all trained model files.

    # Grid search
    grid_search_type : {"exhaustive", "halving", "optuna"}, default: "optuna"
        Type of grid search to apply.
    grid_search_timeout : int, default: 3600
        Time budget for grid search (in seconds).
    n_grid_searches : int, default: 3
        Run grid search for the best `n_grid_searches` (model, feature set) pairs from
        initial modelling.
    n_trials_per_grid_search : int, default: 250
        Maximal number of trials/candidates for each grid search.

    # Stacking of models
    stacking : bool, default: False
        Whether to create a stacking model at the end.

    # Production
    preprocess_function : str, optional, default: None
        Add custom code for the prediction function - useful for production.
        Will be executed with `exec`, can be multi-line, uses data as input.

    # Flags
    plot_eda : bool, default: False
        Whether to run exploratory data analysis.
    process_data : bool, default: True
        Whether to force data processing.
    no_dirs : bool, default: False
        Whether to create files.

    # Other
    kwargs: Any
        Swallows all arguments that are not accepted. Warnings are raised if not empty.
    """

    def __init__(
        self,
        # Main settings
        main_dir: str = "Auto_ML/",
        target: str = None,
        name: str = "AutoML",
        version: int = None,
        mode: str = None,
        objective: str = None,
        verbose: int = 1,
        *,
        # Data processing
        int_cols: list[str] = None,
        float_cols: list[str] = None,
        date_cols: list[str] = None,
        cat_cols: list[str] = None,
        missing_values: str = "zero",
        outlier_removal: str = "clip",
        z_score_threshold: int = 4,
        include_output: bool = False,
        # Balancing
        balance: bool = False,
        # Feature processing
        extract_features: bool = True,
        information_threshold: float = 0.999,
        feature_timeout: int = 3600,
        # Interval analysis
        interval_analyse: bool = True,
        # Sequencing
        sequence: bool = False,
        seq_back: int | list[int] = 1,
        seq_forward: int | list[int] = 1,
        seq_shift: int = 0,
        seq_diff: str = "none",
        seq_flat: bool = True,
        # Modelling
        standardize: bool = False,
        cv_shuffle: bool = True,
        cv_splits: int = 10,
        store_models: bool = False,
        # Grid search
        grid_search_type: str = "optuna",
        grid_search_timeout: int = 3600,
        n_grid_searches: int = 3,
        n_trials_per_grid_search: int = 250,
        # Stacking
        stacking: bool = False,
        # Production
        preprocess_function: str = None,
        # Flags
        plot_eda: bool = False,
        process_data: bool = True,
        no_dirs: bool = False,
        # Other
        **kwargs,
    ):
        # Get init parameters for `self.settings`
        sig, init_locals = signature(self.__init__), locals()
        init_params = {
            param.name: init_locals[param.name] for param in sig.parameters.values()
        }
        del sig, init_locals

        # super() calls
        super().__init__(verbose=verbose)

        # Input checks: validity
        if mode not in (None, "regression", "classification"):
            raise ValueError("Supported models: {'regression', 'classification', None}")
        if not 0 < information_threshold < 1:
            raise ValueError("Information threshold must be within (0, 1) interval.")
        valid_gs_types = {"exhaustive", "halving", "optuna"}
        if (
            grid_search_type is not None
            and grid_search_type.lower() not in valid_gs_types
        ):
            raise ValueError(f"Grid search type must be one of: {valid_gs_types}")

        # Warn unused parameters
        if kwargs:
            warnings.warn(
                f"Got unexpected keyword arguments that are not handled: {set(kwargs)}"
            )

        # Input checks: advices
        if include_output and not sequence:
            warnings.warn(
                "It is strongly advised NOT to include output without sequencing."
            )

        # Main settings
        self.main_dir = f"{Path(main_dir)}/"  # assert backslash afterwards
        self.target = target
        self.name = name
        self.version = version
        self.mode = mode
        self.objective = objective

        # Data processing
        self.int_cols = int_cols
        self.float_cols = float_cols
        self.date_cols = date_cols
        self.cat_cols = cat_cols
        self.missing_values = missing_values
        self.outlier_removal = outlier_removal
        self.z_score_threshold = z_score_threshold
        self.include_output = include_output

        # Balancing
        self.balance = balance

        # Feature processing
        self.extract_features = extract_features
        self.information_threshold = information_threshold
        self.feature_timeout = feature_timeout

        # Interval analysis
        self.use_interval_analyser = interval_analyse

        # Sequencing
        self.sequence = sequence
        self.sequence_back = seq_back
        self.sequence_forward = seq_forward
        self.sequence_shift = seq_shift
        self.sequence_diff = seq_diff
        self.sequence_flat = seq_flat

        # Modelling
        self.standardize = standardize
        self.cv_shuffle = cv_shuffle
        self.cv_splits = cv_splits
        self.store_models = store_models

        # Grid search
        self.grid_search_type = grid_search_type
        self.grid_search_timeout = grid_search_timeout
        self.n_grid_searches = n_grid_searches
        self.n_trials_per_grid_search = n_trials_per_grid_search

        # Stacking
        self.stacking = stacking

        # Production
        self.preprocess_function = preprocess_function

        # Flags
        self.plot_eda = plot_eda
        self.process_data = process_data
        self.no_dirs = no_dirs

        # Create directory
        if not self.no_dirs:
            self._create_dirs()
            self._load_version()
        else:
            self.version = 1

        # Store Pipeline Settings
        self.settings = {"pipeline": init_params}

        # Objective & Scorer
        if self.objective is not None:
            if not isinstance(self.objective, str):
                raise ValueError("Objective needs to be a string.")
            self.scorer = metrics.get_scorer(self.objective)
        else:
            self.scorer = None

        # Required sub-classes
        self.data_sampler = DataSampler()
        self.data_processor = DataProcessor()
        self.data_sequencer = Sequencer()
        self.feature_processor = FeatureProcessor()
        self.interval_analyser = IntervalAnalyser()
        self.drift_detector = DriftDetector()

        # Instance initiating
        self.best_model = None
        self.best_model_str = None
        self.best_feature_set = None
        self.best_score = None
        self._data = None
        self.feature_sets = None
        self.results = None
        self.n_classes = None
        self.is_fitted = False

        # Monitoring
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
            settings_path = self.main_dir + f"Production/v{self.version}/Settings.json"
            assert Path(
                settings_path
            ).exists(), "Cannot load settings from nonexistent version"
            return json.load(open(settings_path, "r"))

    def load_settings(self, settings: dict):
        """
        Restores a pipeline from settings.

        Parameters
        ----------
        settings [dict]: Pipeline settings
        """
        # Set parameters
        settings["pipeline"]["no_dirs"] = True
        self.__init__(**settings["pipeline"])
        self.settings = settings
        self.data_processor.load_settings(settings["data_processing"])
        self.feature_processor.load_settings(settings["feature_processing"])
        # TODO: load_settings for IntervalAnalyser (not yet implemented)
        if "drift_detector" in settings:
            self.drift_detector = DriftDetector(
                num_cols=self.data_processor.float_cols + self.data_processor.int_cols,
                cat_cols=self.data_processor.cat_cols,
                date_cols=self.data_processor.date_cols,
            ).load_weights(settings["drift_detector"])

    def load_model(self, model: object):
        """
        Restores a trained model
        """
        assert type(model).__name__ == self.settings["model"]
        self.best_model = model
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
        self.logger.info(f"\n\n*** Starting Amplo AutoML - {self.name} ***\n\n")

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

        # Detect mode (classification / regression)
        self._mode_detector()

        # Preprocess Data
        self._data_processing()

        # Fit Drift Detector to input
        num_cols = list(
            set(self.x).intersection(
                self.data_processor.float_cols + self.data_processor.int_cols
            )
        )
        date_cols = list(set(self.x).intersection(self.data_processor.date_cols))
        cat_cols = list(set(self.x) - set(num_cols + date_cols))
        self.drift_detector = DriftDetector(
            num_cols=num_cols, cat_cols=cat_cols, date_cols=date_cols
        )
        self.drift_detector.fit(self.x)

        # Run Exploratory Data Analysis
        self._eda()

        # Balance data
        self._data_sampling()

        # Sequence
        self._sequencing()

        # Interval-analyze data
        self._interval_analysis()

        # Extract and select features
        self._feature_processing()

        # Standardize
        # Standardizing assures equal scales, equal gradients and no clipping.
        # Therefore, it needs to be after sequencing & feature processing, as this
        # alters scales
        self._standardizing()

    def model_training(self, **kwargs):
        """Train models

        1. Initial Modelling
            Runs various models with default parameters for all feature sets
            If Sequencing is enabled, this is where it happens, as here, the feature
            set is generated.
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

    def conclude_fitting(self, *, model=None, feature_set=None, params=None):
        """
        Prepare production files that are necessary to deploy a specific
        model / feature set combination

        Creates or modifies the following files
            - ``Model.joblib`` (production model)
            - ``Settings.json`` (model settings)

        Parameters
        ----------
        model : str or list of str, optional
            Model file for which to prepare production files.
        feature_set : str or list of str, optional
            Feature set for which to prepare production files.
        params : dict, optional
            Model parameters for which to prepare production files.
            Default: takes the best parameters
        """
        if not self.no_dirs:

            # Set up production path
            prod_dir = self.main_dir + f"Production/v{self.version}/"
            Path(prod_dir).mkdir(exist_ok=True)

            # Parse arguments
            self._parse_production_args(model, feature_set, params)

            # Verbose printing
            self.logger.info(
                f"Preparing Production files for {self.best_model_str}, "
                f"{self.best_feature_set}"
            )

            # Set best model
            self._prepare_production_model(prod_dir + "Model.joblib")

            # Set and store production settings
            self._prepare_production_settings(prod_dir + "Settings.json")

            # Check data
            obs = DataObserver(pipeline=self)
            obs.observe()
            self.settings["data_observer"] = obs.observations

            # Observe production
            obs = ModelObserver(pipeline=self)
            obs.observe()
            self.settings["model_observer"] = obs.observations

        # Finish
        self.is_fitted = True
        self.logger.info("All done :)")
        utils.logging.remove_file_handler()

    def convert_data(
        self, x: pd.DataFrame, preprocess: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
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
        if self.preprocess_function is not None and preprocess:
            ex_globals = {"data": x}
            exec(self.preprocess_function, ex_globals)
            x = ex_globals["data"]

        # Process data
        x = self.data_processor.transform(x)

        # Drift Check
        self.drift_detector.check(x)

        # Split output
        y = None
        if self.target in x.keys():
            y = x[self.target]
            if not self.include_output:
                x = x.drop(self.target, axis=1)

        # Sequence
        if self.sequence:
            x, y = self.data_sequencer.convert(x, y)

        # Convert Features
        x = self.feature_processor.transform(
            x, feature_set=self.settings["feature_set"]
        )

        # Standardize
        if self.standardize:
            x, y = self._transform_standardize(x, y)

        # NaN test -- datetime should be taken care of by now
        if (
            x.astype("float32").replace([np.inf, -np.inf], np.nan).isna().sum().sum()
            != 0
        ):
            raise ValueError(
                f"Column(s) with NaN: {list(x.keys()[x.isna().sum() > 0])}"
            )

        # Return
        return x, y

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Full script to make predictions. Uses 'Production' folder with defined or
        latest version.

        Parameters
        ----------
        data [pd.DataFrame]: data to do prediction on
        """
        start_time = time.time()
        assert self.is_fitted, "Pipeline not yet fitted."

        # Print
        self.logger.info(
            f"Predicting with {type(self.best_model).__name__}, v{self.version}"
        )

        # Convert
        x, y = self.convert_data(data)

        # Predict
        predictions = self.best_model.predict(x)
        if self.mode == "regression" and self.standardize:
            predictions = self._inverse_standardize(predictions)
        elif self.mode == "classification":
            predictions[:] = self.data_processor.decode_labels(
                predictions.astype(int), except_not_fitted=False
            )

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
        assert (
            self.mode == "classification"
        ), "Predict_proba only available for classification"
        assert hasattr(
            self.best_model, "predict_proba"
        ), f"{type(self.best_model).__name__} has no attribute predict_proba"

        # Logging
        self.logger.info(
            f"Predicting with {type(self.best_model).__name__}, v{self.version}"
        )

        # Convert data
        x, y = self.convert_data(data)

        # Predict
        prediction = self.best_model.predict_proba(x)

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
        kwargs : dict
            Not used, but necessary as we don't split .fit() kwargs

        Returns
        -------
        Pipeline
        """
        assert x is not None or data is not None, "No data provided"
        assert (x is not None) ^ (
            data is not None
        ), "Setting both, `x` and `data`, is ambiguous"

        # Labels are provided separately
        if y is not None:
            # Check data
            x = x if x is not None else data
            assert x is not None, "Parameter ``x`` is not set"
            assert isinstance(
                x, (np.ndarray, pd.Series, pd.DataFrame)
            ), "Unsupported data type for parameter ``x``"
            assert isinstance(
                y, (np.ndarray, pd.Series)
            ), "Unsupported data type for parameter ``y``"

            # Set target manually if not defined
            if not self.target:
                self.target = "target"

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
            assert all(x.index == y.index), "``x`` and ``y`` indices do not match"
            if self.target in x.columns:
                if any(x[self.target] != y):
                    msg = (
                        f"Target column coexists in both, x and y data, but have "
                        f"unequal content. Rename the column ``{self.target}`` in x."
                    )
                    raise ValueError(msg)
                data = x
            else:
                # Concatenate x and y
                data = pd.concat([x, y], axis=1)

        # Set data parameter in case it is provided through parameter ``x``
        data = data if data is not None else x
        metadata = None

        # A path was provided to read out (multi-indexed) data
        if isinstance(data, (str, Path)):
            # Set target manually if not defined
            if not self.target:
                self.target = "target"
            # Parse data
            data, metadata = utils.io.merge_logs(data, self.target)

        # Test data
        assert self.target, "No target string provided"
        assert self.target in data.columns, "Target column missing"
        assert (
            len(data.columns) == data.columns.nunique()
        ), "Columns have no unique names"

        # Save data
        clean_target = utils.clean_feature_name(self.target)
        data.rename(columns={self.target: clean_target}, inplace=True)
        self.target = clean_target
        self._set_data(data)

        # Store metadata in settings
        self.settings["file_metadata"] = metadata or dict()

        return self

    def has_new_training_data(self):
        # Return True if no previous version exists
        if self.version == 1:
            return True

        # Get previous and current file metadata
        curr_metadata = self.settings["file_metadata"]
        last_metadata = self.get_settings(self.version - 1)["file_metadata"]

        # Check each settings file
        for file_id in curr_metadata:
            # Get file specific metadata
            curr = curr_metadata[file_id]
            last = last_metadata.get(file_id, dict())
            # Compare metadata
            same_folder = curr["folder"] == last.get("folder")
            same_file = curr["file"] == last.get("file")
            same_mtime = curr["last_modified"] == last.get("last_modified")
            if not all([same_folder, same_file, same_mtime]):
                return False

        return True

    def _mode_detector(self):
        """
        Detects the mode (Regression / Classification)
        """
        # Only run if mode is not provided
        if self.mode is not None:
            return

        # Classification if string
        if self.y.dtype == str or self.y.nunique() < 0.1 * len(self.data):
            self.mode = "classification"
            self.objective = self.objective or "neg_log_loss"

        # Else regression
        else:
            self.mode = "regression"
            self.objective = self.objective or "neg_mean_absolute_error"

        # Set scorer
        self.scorer = metrics.get_scorer(self.objective)

        # Copy to settings
        self.settings["pipeline"]["mode"] = self.mode
        self.settings["pipeline"]["objective"] = self.objective

        # Logging
        self.logger.info(
            f"Setting mode to {self.mode} & objective to {self.objective}."
        )

    def _data_processing(self):
        """
        Organises the data cleaning. Heavy lifting is done in self.dataProcessor,
        but settings etc. needs to be organised.
        """
        self.data_processor = DataProcessor(
            target=self.target,
            int_cols=self.int_cols,
            float_cols=self.float_cols,
            date_cols=self.date_cols,
            cat_cols=self.cat_cols,
            include_output=True,
            missing_values=self.missing_values,
            outlier_removal=self.outlier_removal,
            z_score_threshold=self.z_score_threshold,
        )

        # Set paths
        extracted_data_path = self.main_dir + f"Data/Extracted_v{self.version}.csv"
        data_path = self.main_dir + f"Data/Cleaned_v{self.version}.csv"
        settings_path = self.main_dir + f"Settings/Cleaning_v{self.version}.json"

        # Only load settings if feature processor is done already.
        if Path(extracted_data_path).exists() and Path(settings_path).exists():
            self.settings["data_processing"] = json.load(open(settings_path, "r"))
            self.data_processor.load_settings(self.settings["data_processing"])

        # Else, if data processor is done, load settings & data
        elif Path(data_path).exists() and Path(settings_path).exists():
            # Load data
            data = self._read_csv(data_path)
            self._set_data(data)

            # Load settings
            self.settings["data_processing"] = json.load(open(settings_path, "r"))
            self.data_processor.load_settings(self.settings["data_processing"])

            self.logger.info("Loaded Cleaned Data")

        # Else, run the data processor
        else:
            # Cleaning
            data = self.data_processor.fit_transform(self.data)

            # Update pipeline
            self._set_data(data)

            # Store data
            self._write_csv(self.data, data_path)

            # Clean up Extracted previous version
            if os.path.exists(
                self.main_dir + f"Data/Extracted_v{self.version - 1}.csv"
            ):
                os.remove(self.main_dir + f"Data/Extracted_v{self.version - 1}.csv")

            # Save settings
            self.settings["data_processing"] = self.data_processor.get_settings()
            if not self.no_dirs:
                json.dump(self.settings["data_processing"], open(settings_path, "w"))

        # If no columns were provided, load them from data processor
        if self.date_cols is None:
            self.date_cols = self.settings["data_processing"]["date_cols"]
        if self.int_cols is None:
            self.int_cols = self.settings["data_processing"]["int_cols"]
        if self.float_cols is None:
            self.float_cols = self.settings["data_processing"]["float_cols"]
        if self.cat_cols is None:
            self.cat_cols = self.settings["data_processing"]["cat_cols"]

        # Assert classes in case of classification
        if self.mode == "classification":
            self.n_classes = self.y.nunique()
            if self.n_classes >= 50:
                warnings.warn(
                    "More than 20 classes, "
                    "you may want to reconsider classification mode"
                )
            if set(self.y) != set([i for i in range(len(set(self.y)))]):
                raise ValueError("Classes should be [0, 1, ...]")

    def _eda(self):
        if not self.plot_eda:
            return

        self.logger.info("Starting Exploratory Data Analysis")
        eda = DataExplorer(
            self.x,
            y=self.y,
            mode=self.mode,
            folder=self.main_dir,
            version=self.version,
        )
        eda.run()

    def _data_sampling(self):
        """
        Only run for classification problems. Balances the data using imblearn.
        Does not guarantee to return balanced classes. (Methods are data dependent)
        """
        if not self.balance or not self.mode == "classification":
            return

        self.data_sampler = DataSampler(
            method="both",
            margin=0.1,
            cv_splits=self.cv_splits,
            shuffle=self.cv_shuffle,
            fast_run=False,
            objective=self.objective,
        )
        data_path = self.main_dir + f"Data/Balanced_v{self.version}.csv"

        if Path(data_path).exists():
            # Load data
            data = self._read_csv(data_path)
            self._set_data(data)

            self.logger.info("Loaded Balanced data")

        else:
            # Fit and resample
            self.logger.info("Resampling data")
            x, y = self.data_sampler.fit_resample(self.x, self.y)

            # Store
            self._set_xy(x, y)
            self._write_csv(self.data, data_path)

    def _sequencing(self):
        """
        Sequences the data. Useful mostly for problems where older samples play a role
        in future values. The settings of this module are NOT AUTOMATIC
        """
        if not self.sequence:
            return

        self.data_sequencer = Sequencer(
            back=self.sequence_back,
            forward=self.sequence_forward,
            shift=self.sequence_shift,
            diff=self.sequence_diff,
        )
        data_path = self.main_dir + f"Data/Sequence_v{self.version}.csv"

        if Path(data_path).exists():
            # Load data
            data = self._read_csv(data_path)
            self._set_data(data)

            self.logger.info("Loaded Extracted Features")

        else:
            # Sequencing
            self.logger.info("Sequencing data")
            x, y = self.data_sequencer.convert(self.x, self.y)

            # Store
            self._set_xy(x, y)
            self._write_csv(self.data, data_path)

    def _feature_processing(self):
        """
        Organises feature processing. Heavy lifting is done in self.featureProcessor,
        but settings, etc. needs to be organised.
        """
        self.feature_processor = FeatureProcessor(
            mode=self.mode,
            is_temporal=None,
            extract_features=self.extract_features,
            collinear_threshold=self.information_threshold,
            verbose=self.verbose,
        )

        # Set paths
        data_path = self.main_dir + f"Data/Extracted_v{self.version}.csv"
        settings_path = self.main_dir + f"Settings/Extracting_v{self.version}.json"

        if Path(data_path).exists() and Path(settings_path).exists():
            # Loading data
            data = self._read_csv(data_path)
            self._set_data(data)

            # Loading settings
            self.settings["feature_processing"] = json.load(open(settings_path, "r"))
            self.feature_processor.load_settings(self.settings["feature_processing"])
            self.feature_sets = self.settings["feature_processing"]["feature_sets_"]

            self.logger.info("Loaded Extracted Features")

        else:
            self.logger.info("Starting Feature Processor")

            # Transform data.  Note that y also needs to be transformed in the
            # case when we're using the temporal feature processor (pooling).
            x = self.feature_processor.fit_transform(self.x, self.y)
            y = self.feature_processor.transform_target(self.y)
            self.feature_sets = self.feature_processor.feature_sets_

            # Store data
            self._set_xy(x, y)
            self._write_csv(self.data, data_path)

            # Cleanup Cleaned_vx.csv
            if os.path.exists(self.main_dir + f"Data/Cleaned_v{self.version}.csv"):
                os.remove(self.main_dir + f"Data/Cleaned_v{self.version}.csv")

            # Save settings
            self.settings["feature_processing"] = self.feature_processor.get_settings()
            if not self.no_dirs:
                json.dump(self.settings["feature_processing"], open(settings_path, "w"))

    def _interval_analysis(self):
        """
        Interval-analyzes the data using ``amplo.auto_ml.interval_analysis``
        or resorts to pre-computed data, if present.
        """
        # Skip analysis when analysis is not possible and/or not desired
        is_multi_index = len(self.x.index.names) == 2
        if not all(
            [self.use_interval_analyser, is_multi_index, self.mode == "classification"]
        ):
            return

        self.interval_analyser = IntervalAnalyser(target=self.target)

        # Set paths
        data_path = self.main_dir + f"Data/Interval_Analyzed_v{self.version}.csv"
        # settings_path = self.main_dir +
        # f'Settings/Interval_Analysis_v{self.version}.json'

        if Path(data_path).exists():  # TODO: and Path(settings_path).exists():
            # Load data
            data = self._read_csv(data_path)
            self._set_data(data)

            # TODO implement `IntervalAnalyser.load_settings` and add to
            #  `self.load_settings`
            # # Load settings
            # self.settings['interval_analysis'] = json.load(open(settings_path, 'r'))
            # self.intervalAnalyser.load_settings(self.settings['interval_analysis'])

            self.logger.info("Loaded interval-analyzed data")

        else:
            self.logger.info("Interval-analyzing data")

            # Transform data
            data = self.interval_analyser.fit_transform(self.x, self.y)

            # Store data
            self._set_data(data)
            self._write_csv(self.data, data_path)

            # TODO implement `IntervalAnalyser.get_settings` and add to
            #  `self.get_settings`
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
        settings_path = self.main_dir + f"Settings/Standardize_v{self.version}.json"

        if Path(settings_path).exists():
            # Load data
            self.settings["standardize"] = json.load(open(settings_path, "r"))

        else:
            # Fit data
            self._fit_standardize(self.x, self.y)

            # Store Settings
            json.dump(self.settings["standardize"], open(settings_path, "w"))

        # Transform data
        x, y = self._transform_standardize(self.x, self.y)
        self._set_xy(x, y)

    def _initial_modelling(self):
        """
        Runs various models to see which work well.
        """

        # Set paths
        results_path = Path(self.main_dir) / "Results.csv"

        # Load existing results
        if results_path.exists():

            # Load results
            self.results = pd.read_csv(results_path)

            # Printing here as we load it
            results = self.results[
                np.logical_and(
                    self.results["version"] == self.version,
                    self.results["type"] == "Initial modelling",
                )
            ]
            for fs in set(results["dataset"]):
                self.logger.info(
                    f"Initial Modelling for {fs} ({len(self.feature_sets[fs])})"
                )
                fsr = results[results["dataset"] == fs]
                for i in range(len(fsr)):
                    row = fsr.iloc[i]
                    self.logger.info(
                        f'{row["model"].ljust(40)} {self.objective}: '
                        f'{row["mean_objective"]:.4f} \u00B1 {row["std_objective"]:.4f}'
                    )

        # Check if this version has been modelled
        if self.results is None or self.version not in self.results["version"].values:

            # Iterate through feature sets
            for feature_set, cols in self.feature_sets.items():

                # Skip empty sets
                if len(cols) == 0:
                    self.logger.info(f"Skipping {feature_set} features, empty set")
                    continue
                self.logger.info(
                    f"Initial Modelling for {feature_set} features ({len(cols)})"
                )

                # Do the modelling
                modeller = Modeller(
                    mode=self.mode,
                    store_models=self.store_models,
                    cv=self.cv,
                    objective=self.objective,
                    dataset=feature_set,
                    store_results=False,
                    folder=self.main_dir + "Models/",
                )
                results = modeller.fit(self.x[cols], self.y)

                # Add results to memory
                results["type"] = "Initial modelling"
                results["version"] = self.version
                if self.results is None:
                    self.results = results
                else:
                    self.results = pd.concat([self.results, results])

            # Save results
            self._write_csv(self.results, results_path)

    def grid_search(self, model=None, feature_set=None, parameter_set=None):
        """
        Runs a grid search.

        By default, takes ``self.results`` and runs for the top ``n =
        self.n_grid_searches`` optimizations. There is the option to provide ``model``
        and ``feature_set``, but **both** have to be provided. In this case, the model
        and dataset combination will be optimized.

        Implemented types, Exhaustive, Halving, Optuna.

        Parameters
        ----------
        model : list of (str or object) or object or str, optional
            Which model to run grid search for.
        feature_set : list of str or str, optional
            Which feature set to run gid search for. Must be provided when `model` is
            not None.
            Options: {rf_threshold, rf_increment, shap_threshold, shap_increment}
        parameter_set : dict, optional
            Parameter grid to optimize over.

        Notes
        -----
        When both parameters, ``model`` and ``feature_set``, are provided, the grid
        search behaves as follows:
            - When both parameters are either of dtype ``str`` or have the same length,
              then grid search will treat them as pairs.
            - When one parameter is an iterable and the other parameter is either a
              string or an iterable of different length, then grid search will happen
              for each unique combination of these parameters.
        """

        # Skip grid search and set best initial model as best grid search parameters
        if self.grid_search_type is None or self.n_grid_searches == 0:
            best_initial_model = self._sort_results(
                self.results[self.results["version"] == self.version]
            ).iloc[:1]
            best_initial_model["type"] = "Hyper Parameter"
            self.results = pd.concat(
                [self.results, best_initial_model], ignore_index=True
            )
            return self

        # Define models
        if model is None:
            # Run through first best initial models (n = self.n_grid_searches)
            selected_results = self.sort_results(
                self.results[
                    np.logical_and(
                        self.results["type"] == "Initial modelling",
                        self.results["version"] == self.version,
                    )
                ]
            ).iloc[: self.n_grid_searches]
            models = [
                utils.get_model(model_name, mode=self.mode, samples=len(self.x))
                for model_name in selected_results["model"]
            ]
            feature_sets = selected_results["dataset"]

        elif feature_set is None:
            raise AttributeError(
                "When `model` is provided, `feature_set` cannot be None. "
                "Provide either both params or neither of them."
            )

        else:
            models = (
                [utils.get_model(model, mode=self.mode, samples=len(self.x))]
                if isinstance(model, str)
                else [model]
            )
            feature_sets = (
                [feature_set] if isinstance(feature_set, str) else list(feature_set)
            )
            if len(models) != len(feature_sets):
                # Create each combination
                combinations = list(
                    itertools.product(np.unique(models), np.unique(feature_sets))
                )
                models = [elem[0] for elem in combinations]
                feature_sets = [elem[1] for elem in combinations]

        # Iterate and grid search over each pair of model and feature_set
        for model, feature_set in zip(models, feature_sets):

            # Organise existing model results
            m_results = self.results[
                np.logical_and(
                    self.results["model"] == type(model).__name__,
                    self.results["version"] == self.version,
                )
            ]
            m_results = self._sort_results(
                m_results[m_results["dataset"] == feature_set]
            )

            # Skip grid search if optimized model already exists
            if not ("Hyper Parameter" == m_results["type"]).any():
                # Run grid search for model
                grid_search_results = self._grid_search_iteration(
                    model, parameter_set, feature_set
                )
                grid_search_results = self.sort_results(grid_search_results)

                # Store results
                grid_search_results["version"] = self.version
                grid_search_results["dataset"] = feature_set
                grid_search_results["type"] = "Hyper Parameter"
                self.results = pd.concat(
                    [self.results, grid_search_results], ignore_index=True
                )
                self.results.to_csv(self.main_dir + "Results.csv", index=False)

        return self

    def _create_stacking(self):
        """
        Based on the best performing models, in addition to cheap models based on very
        different assumptions, a stacking model is optimized to enhance/combine the
        performance of the models.
        --> should contain a large variety of models
        --> classifiers need predict_proba
        --> level 1 needs to be ordinary least squares
        """
        if not self.stacking:
            return

        self.logger.info("Creating Stacking Ensemble")

        # Select most frequent feature set from hyperparameter optimization
        results = self._sort_results(
            self.results[
                np.logical_and(
                    self.results["type"] == "Hyper Parameter",
                    self.results["version"] == self.version,
                )
            ]
        )
        feature_set = results["dataset"].value_counts().index[0]
        results = results[results["dataset"] == feature_set]
        self.logger.info("Selected Stacking feature set: {}".format(feature_set))

        # Create Stacking Model Params
        n_stacking_models = 3
        stacking_models_str = results["model"].unique()[:n_stacking_models]
        stacking_models_params = [
            utils.io.parse_json(
                results.iloc[np.where(results["model"] == sms)[0][0]]["params"]
            )
            for sms in stacking_models_str
        ]
        stacking_models = dict(
            [
                (sms, stacking_models_params[i])
                for i, sms in enumerate(stacking_models_str)
            ]
        )
        self.logger.info("Stacked models: {}".format(list(stacking_models.keys())))
        # Make the dict of dict flat
        add_to_stack = list(stacking_models)
        stacking_models = {
            f"{model}__{param}": parameters[param]
            for model, parameters in stacking_models.items()
            for param in parameters
        }

        # Add samples & Features
        stacking_models["n_samples"], stacking_models["n_features"] = self.x.shape

        # Prepare Stack
        if self.mode == "regression":
            stack = StackingRegressor(add_to_stack, **stacking_models)
        elif self.mode == "classification":
            stack = StackingClassifier(add_to_stack, **stacking_models)
        else:
            raise NotImplementedError("Unknown mode")

        # Cross Validate
        x, y = self.x[self.feature_sets[feature_set]].to_numpy(), self.y.to_numpy()
        score = []
        times = []
        for (t, v) in tqdm(self.cv.split(x, y)):
            start_time = time.time()
            xt, xv, yt, yv = x[t], x[v], y[t].reshape((-1)), y[v].reshape((-1))
            model = copy.deepcopy(stack)
            model.fit(xt, yt)
            score.append(self.scorer(model, xv, yv))
            times.append(time.time() - start_time)

        # Output Results
        self.logger.info("Stacking result:")
        self.logger.info(
            f"{self.objective}:     {np.mean(score):.2f} \u00B1 {np.std(score):.2f}"
        )
        self.results = self.results.append(
            {
                "date": datetime.today().strftime("%d %b %y"),
                "model": type(stack).__name__,
                "dataset": feature_set,
                "params": json.dumps(stack.get_params()),
                "mean_objective": np.mean(score),
                "std_objective": np.std(score),
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "version": self.version,
                "type": "Stacking",
            },
            ignore_index=True,
        )
        self.results.to_csv(self.main_dir + "Results.csv", index=False)

    def _parse_production_args(self, model=None, feature_set=None, params=None):
        """
        Parse production arguments. Selects the best model, feature set and parameter
        combination.

        Parameters
        ----------
        model : str or list of str, optional
            Model constraint(s). In case list is provided, the pipeline needs to be
            fitted.
        feature_set : str or list of str, optional
            Feature set constraint(s). In case list is provided, the pipeline needs to
            be fitted.
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
        if self.results is None and (
            model is None or params is None or feature_set is None
        ):
            raise ValueError(
                "Pipeline not fitted and no model, params or feature set provided."
            )

        if model is not None and not isinstance(model, str):
            # TODO: This issue is linked with AML-103 (in Jira)
            #  1. Add to method docstring that it accepts a model instance, too
            #  2. Change `if`-case to a single `isinstance(model, BasePredictor)`
            # Get model name
            model = type(model).__name__

        # Get results of current version
        results = (
            self._sort_results(self.results[self.results["version"] == self.version])
            if self.results is not None
            else None
        )

        # Filter results for model
        if model is not None and results is not None:
            # Enforce list
            if isinstance(model, str):
                model = [model]

            # Filter results
            results = self._sort_results(results[results["model"].isin(model)])

        # filter results for feature set
        if feature_set is not None and results is not None:
            if isinstance(feature_set, str):
                feature_set = [feature_set]
            # Filter results
            results = self._sort_results(results[results["dataset"].isin(feature_set)])

        # Get best parameters
        if params is None and results is not None:
            params = results.iloc[0]["params"]
        elif params is None:
            params = {}

        # Find the best allowed arguments
        self.best_model_str = model if results is None else results.iloc[0]["model"]
        self.best_feature_set = (
            feature_set if results is None else results.iloc[0]["dataset"]
        )
        self.best_params = utils.io.parse_json(params)
        self.best_score = 0 if results is None else results.iloc[0]["worst_case"]

        return self

    def _prepare_production_model(self, model_path):
        """
        Prepare and store `self.bestModel` for production

        Parameters
        ----------
        model_path : str or Path
            Where to store model for production

        Returns
        -------
        """
        model_path = Path(model_path)

        # Make model
        if "Stacking" in self.best_model_str:
            # Create stacking
            if self.mode == "regression":
                stacking_model = StackingRegressor
            elif self.mode == "classification":
                stacking_model = StackingClassifier
            else:
                raise NotImplementedError("Mode not set")

            self.best_model = stacking_model(
                add_to_stack=self.best_params.get("add_to_stack", None),
                add_defaults_to_stack=self.best_params.get(
                    "add_defaults_to_stack", True
                ),
                n_samples=self.x.shape[0],
                n_features=self.x.shape[1],
            )
        else:
            # Take model as is
            self.best_model = utils.get_model(
                self.best_model_str, mode=self.mode, samples=len(self.x)
            )

        # Set params, train
        self.best_model.set_params(**self.best_params)
        self.best_model.fit(self.x[self.feature_sets[self.best_feature_set]], self.y)

        # Save model
        joblib.dump(self.best_model, model_path)

        self.best_score = self.scorer(
            self.best_model,
            self.x[self.feature_sets[self.best_feature_set]],
            self.y,
        )
        self.logger.info(
            f"Model fully fitted, in-sample {self.objective}: {self.best_score:4f}"
        )

        return

    def _prepare_production_settings(self, settings_path):
        """
        Prepare `self.settings` for production and dump to file

        Parameters
        ----------
        settings_path : str or Path
            Where to save settings for production
        """
        assert self.best_model is not None, "`self.bestModel` is not yet prepared"
        settings_path = Path(settings_path)

        # Update pipeline settings
        self.settings["version"] = self.version
        self.settings["pipeline"]["verbose"] = self.verbose
        self.settings["model"] = self.best_model_str
        self.settings["params"] = self.best_params
        self.settings["feature_set"] = self.best_feature_set
        self.settings["features"] = self.feature_sets[self.best_feature_set]
        self.settings["best_score"] = self.best_score
        self.settings["amplo_version"] = (
            amplo.__version__ if hasattr(amplo, "__version__") else "dev"
        )

        # Validation
        validator = ModelValidator(
            cv_splits=self.cv_splits,
            cv_shuffle=self.cv_shuffle,
            verbose=self.verbose,
        )
        self.settings["validation"] = validator.validate(
            model=self.best_model, x=self.x, y=self.y, mode=self.mode
        )

        # Prune Data Processor
        required_features = self.feature_processor.get_required_columns(
            self.best_feature_set
        )
        self.data_processor.prune_features(required_features)
        self.settings["data_processing"] = self.data_processor.get_settings()

        # Fit Drift Detector to output and store settings
        self.drift_detector.fit_output(
            self.best_model, self.x[self.feature_sets[self.best_feature_set]]
        )
        self.settings["drift_detector"] = self.drift_detector.get_weights()

        # Save settings
        json.dump(self.settings, open(settings_path, "w"), indent=4)

    # Getter Functions / Properties
    @property
    def cv(self):
        """
        Gives the Cross Validation scheme

        Returns
        -------
        cv : KFold or StratifiedKFold
            The cross validator
        """
        # Regression
        if self.mode == "regression":
            return KFold(
                n_splits=self.cv_splits,
                shuffle=self.cv_shuffle,
                random_state=83847939 if self.cv_shuffle else None,
            )

        # Classification
        if self.mode == "classification":
            return StratifiedKFold(
                n_splits=self.cv_splits,
                shuffle=self.cv_shuffle,
                random_state=83847939 if self.cv_shuffle else None,
            )

    @property
    def data(self) -> Union[None, pd.DataFrame]:
        return self._data

    @property
    def x(self) -> pd.DataFrame:
        if self.data is None:
            raise AttributeError("Data is None")
        if self.include_output:
            return self.data
        return self.data.drop(self.target, axis=1)

    @property
    def y(self):
        if self.data is None:
            raise AssertionError("`self.data` is empty. Set a value with `set_data`")
        return self.data[self.target]

    @property
    def y_orig(self):
        enc_labels = self.y
        dec_labels = self.data_processor.decode_labels(
            enc_labels, except_not_fitted=False
        )
        return pd.Series(dec_labels, name=self.target, index=enc_labels.index)

    # Setter Functions
    def _set_data(self, new_data: pd.DataFrame):
        assert isinstance(new_data, pd.DataFrame), "Invalid data type"
        assert self.target in new_data, "No target column present"
        assert len(new_data.columns) > 1, "No feature column present"
        self._data = new_data

    def _set_xy(
        self,
        new_x: Union[np.ndarray, pd.DataFrame],
        new_y: Union[np.ndarray, pd.Series],
    ):
        if not isinstance(new_y, pd.Series):
            new_y = pd.Series(new_y, name=self.target)
        new_data = pd.concat([new_x, new_y], axis=1)
        self._set_data(new_data)

    def _set_x(self, new_x: Union[np.ndarray, pd.DataFrame]):
        # Convert to dataframe
        if isinstance(new_x, np.ndarray):
            old_x = self.data.drop(self.target, axis=1)
            old_x_shape = old_x.shape
            old_x_index = old_x.index
            old_x_columns = old_x.columns
            del old_x

            if new_x.shape == old_x_shape:
                new_x = pd.DataFrame(new_x, index=old_x_index, columns=old_x_columns)
            else:
                warnings.warn(
                    "Old x-data has more/less columns than new x-data. "
                    "Setting dummy feature names..."
                )
                columns = [f"Feature_{i}" for i in range(new_x.shape[1])]
                new_x = pd.DataFrame(new_x, index=old_x_index, columns=columns)

        elif not isinstance(new_x, pd.DataFrame):
            raise ValueError(f"Invalid dtype for new x data: {type(new_x)}")

        # Assert that target is not in x-data
        if self.target in new_x:
            raise AttributeError("Target column name should not be in x-data")

        # Set data
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
        assert Path(data_path).suffix == ".csv", "Expected a *.csv path"

        data = pd.read_csv(data_path)

        if {"index", "log"}.issubset(data.columns):
            # Multi-index: case when IntervalAnalyser was used
            index = ["log", "index"]
        elif "index" in data.columns:
            index = ["index"]
        else:
            raise IndexError(
                "No known index was found. "
                "Expected to find at least a column named `index`."
            )
        return data.set_index(index)

    def _write_csv(self, data, data_path):
        """
        Write data to given path and set index if needed.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
        data_path : str or Path
        """
        assert Path(data_path).suffix == ".csv", "Expected a *.csv path"

        # Set single-index if not already present
        if len(data.index.names) == 1 and data.index.name is None:
            data.index.name = "index"

        # Raise error if unnamed index is present
        if None in data.index.names:
            raise IndexError(
                f"Found an unnamed index column ({list(data.index.names)})."
            )

        # Write data
        if not self.no_dirs:
            data.to_csv(data_path)

    def _load_version(self):
        """
        Upon start, loads version
        """
        # No need if version is set
        if self.version is not None:
            return

        # Find production folders
        completed_versions = len(os.listdir(self.main_dir + "Production"))
        self.version = completed_versions + 1

        self.logger.info(f"Setting Version {self.version}")

    def _create_dirs(self):
        folders = [
            "",
            "EDA",
            "Data",
            "Features",
            "Documentation",
            "Production",
            "Settings",
        ]
        for folder in folders:
            if not os.path.exists(self.main_dir + folder):
                os.makedirs(self.main_dir + folder)

    def sort_results(self, results: pd.DataFrame) -> pd.DataFrame:
        return self._sort_results(results)

    def _fit_standardize(self, x: pd.DataFrame, y: pd.Series):
        """
        Fits a standardization parameters and returns the transformed data
        """

        # Fit Input
        cat_cols = [
            k
            for lst in self.settings["data_processing"]["dummies"].values()
            for k in lst
        ]
        features = [
            k for k in x.keys() if k not in self.date_cols and k not in cat_cols
        ]
        means_ = x[features].mean(axis=0)
        stds_ = x[features].std(axis=0)
        stds_[stds_ == 0] = 1
        settings = {
            "input": {
                "features": features,
                "means": means_.to_list(),
                "stds": stds_.to_list(),
            }
        }

        # Fit Output
        if self.mode == "regression":
            std = y.std()
            settings["output"] = {
                "mean": y.mean(),
                "std": std if std != 0 else 1,
            }

        self.settings["standardize"] = settings

    def _transform_standardize(
        self, x: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Standardizes the input and output with values from settings.

        Parameters
        ----------
        x [pd.DataFrame]: Input data
        y [pd.Series]: Output data
        """
        # Input
        assert self.settings["standardize"], "Standardize settings not found."

        # Pull from settings
        features = self.settings["standardize"]["input"]["features"]
        means = self.settings["standardize"]["input"]["means"]
        stds = self.settings["standardize"]["input"]["stds"]

        # Filter if not all features are present
        if len(x.keys()) < len(features):
            indices = [
                [i for i, j in enumerate(features) if j == k][0] for k in x.keys()
            ]
            features = [features[i] for i in indices]
            means = [means[i] for i in indices]
            stds = [stds[i] for i in indices]

        # Transform Input
        x[features] = (x[features] - means) / stds

        # Transform output (only with regression)
        if self.mode == "regression":
            y = (y - self.settings["standardize"]["output"]["mean"]) / self.settings[
                "standardize"
            ]["output"]["std"]

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
        assert self.settings["standardize"], "Standardize settings not found"
        return (
            y * self.settings["standardize"]["output"]["std"]
            + self.settings["standardize"]["output"]["mean"]
        )

    @staticmethod
    def _sort_results(results: pd.DataFrame) -> pd.DataFrame:
        return results.sort_values("worst_case", ascending=False)

    def _get_best_params(self, model, feature_set: str) -> dict:
        # Filter results for model and version
        results = self.results[
            np.logical_and(
                self.results["model"] == type(model).__name__,
                self.results["version"] == self.version,
            )
        ]

        # Filter results for feature set & sort them
        results = self._sort_results(results[results["dataset"] == feature_set])

        # Warning for unoptimized results
        if "Hyper Parameter" not in results["type"].values:
            warnings.warn("Hyper parameters not optimized for this combination")

        # Parse & return best parameters (regardless of if it's optimized)
        return utils.io.parse_json(results.iloc[0]["params"])

    def _grid_search_iteration(
        self, model, parameter_set: Union[str, None], feature_set: str
    ):
        """
        INTERNAL | Grid search for defined model, parameter set and feature set.
        """
        self.logger.info(
            f"Starting Hyper Parameter Optimization for {type(model).__name__} on "
            f"{feature_set} features ({len(self.x)} samples, "
            f"{len(self.feature_sets[feature_set])} features)"
        )

        # Select right hyper parameter optimizer
        if self.grid_search_type == "exhaustive":
            grid_search = ExhaustiveGridSearch(
                model,
                params=parameter_set,
                cv=self.cv,
                scoring=self.objective,
                n_trials=self.n_trials_per_grid_search,
                timeout=self.grid_search_timeout,
                verbose=self.verbose,
            )
        elif self.grid_search_type == "halving":
            grid_search = HalvingGridSearch(
                model,
                params=parameter_set,
                cv=self.cv,
                scoring=self.objective,
                n_trials=self.n_trials_per_grid_search,
                verbose=self.verbose,
            )
        elif self.grid_search_type == "optuna":
            grid_search = OptunaGridSearch(
                model,
                timeout=self.grid_search_timeout,
                cv=self.cv,
                n_trials=self.n_trials_per_grid_search,
                scoring=self.objective,
                verbose=self.verbose,
            )
        else:
            raise NotImplementedError(
                "Only Exhaustive, Halving and Optuna are implemented."
            )
        # Get results
        results = grid_search.fit(self.x[self.feature_sets[feature_set]], self.y)
        results = results.sort_values("worst_case", ascending=False)

        # Warn when best hyperparameters are close to predefined grid
        edge_params = grid_search.get_parameter_min_max()
        best_params = pd.Series(results["params"].iloc[0], name="best")
        params = edge_params.join(best_params, how="inner")

        def warn_when_too_close_to_edge(param: pd.Series, tol=0.01):
            # Min-max scaling
            min_, best, max_ = param["min"], param["best"], param["max"]
            scaled = (best - min_) / (max_ - min_)
            # Check if too close and warn if so
            if not (tol < scaled < 1 - tol):
                msg = "Optimal value for parameter is very close to edge case: "
                if min(abs(min_), abs(max_)) < 1:
                    msg += f"{param.name}={best:.2e} (range: {min_:.2e}...{max_:.2e})"
                else:
                    msg += f"{param.name}={best} (range: {min_}...{max_})"
                warnings.warn(msg)

        params.apply(warn_when_too_close_to_edge, axis=1)

        return results

    def _get_main_predictors(self, data):
        """
        Using Shapely Additive Explanations, this function calculates the main
        predictors for a given prediction and sets them into the class' memory.
        """
        # shap.TreeExplainer is not implemented for all models. So we try and fall back
        # to the feature importance given by the feature processor.
        # Note that the error would be raised when calling `TreeExplainer(best_model)`.
        try:
            # Get shap values
            best_model = self.best_model
            if type(best_model).__module__.startswith("amplo"):
                best_model = self.best_model.model
            # Note: The error would be raised at this point.
            #  So we have not much overhead.
            shap_values = np.array(TreeExplainer(best_model).shap_values(data))

            # Average over classes if necessary
            if shap_values.ndim == 3:
                shap_values = np.mean(np.abs(shap_values), axis=0)

            # Average over samples
            shap_values = np.mean(np.abs(shap_values), axis=0)
            shap_values /= shap_values.sum()  # normalize to sum up to 1
            idx_sort = np.flip(np.argsort(shap_values))

            # Set class attribute
            main_predictors = {
                col: score
                for col, score in zip(data.columns[idx_sort], shap_values[idx_sort])
            }

        except Exception:  # the exception can't be more specific  # noqa
            # Get shap feature importance
            fi = self.settings["feature_processing"]["feature_importance_"]["shap"]

            # Use only those columns that are present in the data
            main_predictors = {}
            missing_columns = []
            for col in data:
                if col in fi:
                    main_predictors[col] = fi[col]
                else:
                    missing_columns.append(col)

            if missing_columns:
                self.logger.warning(
                    f"Some data column names are missing in the shap feature "
                    f"importance dictionary: {missing_columns}"
                )

        # Some feature names are obscure since they come from the feature processing
        # module. Here, we relate the feature importance back to the original features.
        translation = self.feature_processor.translate_features(list(main_predictors))
        scores = {}
        for key, features in translation.items():
            for feat in features:
                scores[feat] = scores.get(feat, 0.0) + main_predictors[key]
        # Normalize
        total_score = np.sum(list(scores.values()))
        for key in scores:
            scores[key] /= total_score

        # Set attribute and return
        self._main_predictors = scores
        return scores
