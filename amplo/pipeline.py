#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import json
import os
import time
from inspect import signature
from pathlib import Path
from typing import Any
from warnings import warn

import joblib
import numpy as np
import pandas as pd
from shap import TreeExplainer
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold

import amplo
from amplo import utils
from amplo.automl.data_processing import DataProcessor
from amplo.automl.feature_processing import FeatureProcessor
from amplo.automl.feature_processing.feature_processor import (
    get_required_columns,
    translate_features,
)
from amplo.automl.grid_search import OptunaGridSearch
from amplo.automl.interval_analysis import IntervalAnalyser
from amplo.automl.modelling import Modeller
from amplo.automl.sequencing import Sequencer
from amplo.automl.standardization import Standardizer
from amplo.base import BasePredictor
from amplo.base.objects import LoggingMixin
from amplo.observation import DataObserver, ModelObserver
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
        Main directory of pipeline
    target : str, optional
        Column name of the output variable.
    name : str, default: "AutoML"
        Name of the project
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
    logging_to_file : bool, default: False
        Whether to write logging to a file
    logging_path : str, default: "AutoML.log"
        Write to logging to given path if ``logs_to_file`` is True.

    # Data processing
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
    use_wavelets : bool, default: False
        Whether to use wavelet transforms (useful for frequency data)

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
    grid_search_timeout : int, default: 3600
        Time budget for grid search (in seconds).
    n_grid_searches : int, default: 3
        Run grid search for the best `n_grid_searches` (model, feature set) pairs from
        initial modelling.
    n_trials_per_grid_search : int, default: 250
        Maximal number of trials/candidates for each grid search.

    # Flags
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
        target: str = "target",
        name: str = "AutoML",
        version: int = 1,
        mode: str | None = None,
        objective: str | None = None,
        verbose: int = 1,
        logging_to_file: bool = False,
        logging_path: str | None = None,
        *,
        # Data processing
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
        use_wavelets: bool = False,
        # Interval analysis
        interval_analyse: bool = False,
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
        grid_search_timeout: int = 3600,
        n_grid_searches: int = 3,
        n_trials_per_grid_search: int = 250,
        # Flags
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

        # Initialize Logger
        super().__init__(verbose=verbose)
        if logging_path is None:
            logging_path = f"{Path(main_dir)}/AutoML.log"
        if logging_to_file:
            utils.logging.add_file_handler(logging_path)

        # Input checks: validity
        if mode not in (None, "regression", "classification"):
            raise ValueError("Supported models: {'regression', 'classification', None}")
        if not 0 < information_threshold < 1:
            raise ValueError("Information threshold must be within (0, 1) interval.")

        # Input checks: advices
        if kwargs:
            warn(f"Got unexpected keyword arguments that are not handled: {kwargs}")
        if include_output and not sequence:
            warn("It is strongly advised NOT to include output without sequencing.")
        if interval_analyse and not standardize:
            warn(
                "Data needs to be normalized for the interval analyser, setting "
                "standardize = True."
            )
            standardize = True

        # Main settings
        self.main_dir = f"{Path(main_dir)}/"  # assert backslash afterwards
        self.target = target
        self.name = name
        self.version = version
        self.mode = mode
        self.objective = objective

        # Data processing
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
        self.use_wavelets = use_wavelets

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
        self.grid_search_timeout = grid_search_timeout
        self.n_grid_searches = n_grid_searches
        self.n_trials_per_grid_search = n_trials_per_grid_search

        # Flags
        self.process_data = process_data
        self.no_dirs = no_dirs

        # Set version
        self.version = version if version else 1

        # Store Pipeline Settings
        self.settings: dict[str, Any] = {"pipeline": init_params}

        # Objective & Scorer
        if self.objective is not None:
            if not isinstance(self.objective, str):
                raise ValueError("Objective needs to be a string.")
            self.scorer = metrics.get_scorer(self.objective)
        else:
            self.scorer = None

        # Required sub-classes
        self.data_processor = None
        self.data_sequencer = None
        self.interval_analyser = None
        self.feature_processor = None
        self.standardizer = None

        # Instance initiating
        self.samples_: int | None = None
        self.best_model_: BasePredictor | None = None
        self.best_model_str_: str | None = None
        self.best_params_: dict | None = None
        self.best_feature_set_: str | None = None
        self.best_score_: float | None = None
        self.feature_sets_: dict[str, list[str]] | None = None
        self.results_: pd.DataFrame = pd.DataFrame(
            columns=[
                "feature_set",
                "score",
                "worst_case",
                "date",
                "model",
                "params",
                "time",
            ]
        )
        self.is_fitted_ = False

        # Monitoring
        self._prediction_time_: float | None = None
        self.main_predictors_: dict | None = None

    # User Pointing Functions
    def load(self):
        """
        Restores a pipeline from directory, given main_dir and version.
        """
        assert self.main_dir and self.version

        # Load settings
        settings_path = self.main_dir + "Settings.json"
        with open(settings_path, "r") as settings:
            self.load_settings(json.load(settings))

        # Load model
        model_path = self.main_dir + "Model.joblib"
        self.load_model(joblib.load(model_path))

    def load_settings(self, settings: dict):
        """
        Restores a pipeline from settings.

        Parameters
        ----------
        settings : dict
            Pipeline settings.
        """
        # Set parameters
        settings["pipeline"]["no_dirs"] = True
        settings["pipeline"]["main_dir"] = self.main_dir
        self.__init__(**settings["pipeline"])
        self.settings = settings
        self.best_model_str_ = settings.get("model")
        self.best_params_ = settings.get("params", {})
        self.best_score_ = settings.get("best_score")
        self.best_features_ = settings.get("features")
        self.best_feature_set_ = settings.get("feature_set")
        self.data_processor = DataProcessor().load_settings(settings["data_processing"])
        self.feature_processor = FeatureProcessor().load_settings(
            settings["feature_processing"]
        )
        self.standardizer = Standardizer().load_settings(
            settings.get("standardizing", {})
        )

    def load_model(self, model: BasePredictor):
        """
        Restores a trained model
        """
        assert type(model).__name__ == self.settings["model"]
        self.best_model_ = model
        self.is_fitted_ = True

    def fit(
        self,
        data_or_path: np.ndarray | pd.DataFrame | str | Path,
        target: np.ndarray | pd.Series | str | None = None,
        *,
        metadata: dict[str, dict] | None = None,
        model: str | list[str] | None = None,
        feature_set: str | list[str] | None = None,
    ):
        """
        Fit the full AutoML pipeline.
            1. Prepare data for training
            2. Train / optimize models
            3. Prepare Production Files
                Nicely organises all required scripts / files to make a prediction

        Parameters
        ----------
        data_or_path : np.ndarray or pd.DataFrame or str or Path
            Data or path to data. Propagated to `self.data_preparation`.
        target : np.ndarray or pd.Series or str
            Target data or column name. Propagated to `self.data_preparation`.
        *
        metadata : dict of {int : dict of {str : str or float}}, optional
            Metadata. Propagated to `self.data_preparation`.
        model : str or list of str, optional
            Constrain grid search and fitting conclusion to given model(s).
            Propagated to `self.model_training` and `self.conclude_fitting`.
        feature_set : str or list of str, optional
            Constrain grid search and fitting conclusion to given feature set(s).
            Propagated to `self.model_training` and `self.conclude_fitting`.
            Options: {rf_threshold, rf_increment, shap_threshold, shap_increment}
        params : dict, optional
            Constrain parameters for fitting conclusion.
            Propagated to `self.conclude_fitting`.
        """
        # Starting
        self.logger.info(f"\n\n*** Starting Amplo AutoML - {self.name} ***\n\n")

        # Reading data
        data = self._read_data(data_or_path, target, metadata=metadata)

        # Detect mode (classification / regression)
        self._mode_detector(data)
        self._set_subclasses()
        assert (
            self.mode
            and self.objective
            and self.data_processor
            and self.data_sequencer
            and self.interval_analyser
            and self.feature_processor
            and self.standardizer
        )

        # Preprocess Data
        data = self.data_processor.fit_transform(data)

        # Sequence
        if self.sequence:
            data = self.data_sequencer.fit_transform(data)

        # Extract and select features
        data = self.feature_processor.fit_transform(data)
        self.feature_sets_ = self.feature_processor.feature_sets_

        # Standardize
        if self.standardize:
            data = self.standardizer.fit_transform(data)

        # Interval-analyze data
        if (
            self.use_interval_analyser
            and len(data.index.names) == 2
            and self.mode == "classification"
        ):
            data = self.interval_analyser.fit_transform(data)

        # Model Training #
        ##################
        # TODO: add model limitation
        for feature_set, cols in self.feature_sets_.items():
            self.logger.info(f"Fitting modeller on: {feature_set}")
            feature_data: pd.DataFrame = data[cols + [self.target]]
            results_ = Modeller(
                target=self.target,
                mode=self.mode,
                cv=self.cv,
                objective=self.objective,
                verbose=self.verbose,
            ).fit(feature_data)
            results_["feature_set"] = feature_set
            self.results_ = pd.concat([results_, self.results_])
        self.results_ = self.sort_results_(self.results_)

        # Optimize Hyper parameters
        for model, feature_set in self.grab_grid_search_iterations():
            # TODO: implement models limitations
            assert feature_set in self.feature_sets_
            self.logger.info(
                f"Starting Hyper Parameter Optimization for {type(model).__name__} on "
                f"{feature_set} features ({self.samples_} samples, "
                f"{len(self.feature_sets_[feature_set])} features)"
            )
            results_ = OptunaGridSearch(
                utils.get_model(model, mode=self.mode, samples=self.samples_),
                target=self.target,
                timeout=self.grid_search_timeout,
                cv=self.cv,
                n_trials=self.n_trials_per_grid_search,
                scoring=self.objective,
                verbose=self.verbose,
            ).fit(data)
            self.results_ = pd.concat([self.results_, results_], ignore_index=True)
        self.results_ = self.sort_results_(self.results_)

        # Storing model
        self.store_best(data)

        # Observe
        self.settings["data_observer"] = DataObserver().observe(
            data, self.mode, self.target, self.data_processor.dummies_
        )
        self.settings["model_observer"] = ModelObserver().observe(
            self.best_model_, data, self.target, self.mode  # type: ignore
        )

        # Finish
        self.is_fitted_ = True
        self.logger.info("All done :)")
        utils.logging.del_file_handlers()

    def grab_grid_search_iterations(self):
        iterations = []
        for i in range(self.n_grid_searches):
            row = self.results_.iloc[i]
            iterations.append((row["model"], row["feature_set"]))
        return iterations

    def store_best(self, data: pd.DataFrame):
        # TODO implement models limitations
        assert (
            self.feature_sets_
            and self.scorer
            and self.mode
            and self.data_processor
            and self.data_sequencer
            and self.feature_processor
            and self.standardizer
        )

        # Gather best results_
        self.best_score_ = self.results_.iloc[0]["worst_case"]
        self.best_model_str_ = self.results_.iloc[0]["model"]
        self.best_feature_set_ = self.results_.iloc[0]["feature_set"]
        self.best_features_ = self.feature_sets_.get(self.best_feature_set_, [])
        self.best_params_ = utils.io.parse_json(self.results_.iloc[0]["params"])  # type: ignore

        # Train model on all training data
        self.best_model_ = utils.get_model(
            self.best_model_str_, mode=self.mode, samples=self.samples_
        )
        self.best_model_.set_params(**self.best_params_)
        self.best_model_.fit(data[self.best_features_], data[self.target])

        # Prune Data Processor
        required_features = get_required_columns(
            self.feature_processor.feature_sets_[self.best_feature_set_],
            self.feature_processor.numeric_cols_,
        )
        self.data_processor.prune_features(required_features)

        # Update pipeline settings
        self.settings["version"] = self.version
        self.settings["pipeline"]["verbose"] = self.verbose
        self.settings["model"] = self.best_model_str_
        self.settings["params"] = self.best_params_
        self.settings["feature_set"] = self.best_feature_set_
        self.settings["features"] = self.best_features_
        self.settings["data_processing"] = self.data_processor.get_settings()
        self.settings["feature_processing"] = self.feature_processor.get_settings()
        if self.standardize:
            self.settings["standardizing"] = self.standardizer.get_settings()
        self.settings["best_score"] = self.best_score_
        self.settings["amplo_version"] = (
            amplo.__version__ if hasattr(amplo, "__version__") else "dev"  # type: ignore
        )

        # Validation
        validator = ModelValidator(
            target=self.target,
            cv=self.cv,
            verbose=self.verbose,
        )
        self.settings["validation"] = validator.validate(
            model=self.best_model_, data=data, mode=self.mode
        )

        # Return if no_dirs flag is set
        if self.no_dirs:
            return

        # Create directory
        if not os.path.exists(self.main_dir):
            os.makedirs(self.main_dir)

        # Save model & settings
        joblib.dump(self.best_model_, self.main_dir + "Model.joblib")
        with open(self.main_dir + "Settings.json", "w") as settings:
            json.dump(self.settings, settings, indent=4, cls=utils.io.NpEncoder)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("Pipeline not yet fitted.")
        assert self.data_processor and self.feature_processor and self.standardizer

        # Process data
        data = self.data_processor.transform(data)

        # Sequence
        if self.sequence:
            warn("Sequencer is temporarily disabled.", DeprecationWarning)

        # Convert Features
        data = self.feature_processor.transform(
            data, feature_set=self.settings["feature_set"]
        )

        # Standardize
        if self.standardize:
            data = self.standardizer.transform(data)

        # Output
        if not self.include_output and self.target in data:
            data = data.drop(self.target, axis=1)

        # Return
        return data

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Full script to make predictions. Uses 'Production' folder with defined or
        latest version.

        Parameters
        ----------
        data : pd.DataFrame
            Data to do prediction on.
        """
        start_time = time.time()
        if not self.is_fitted_:
            raise ValueError("Pipeline not yet fitted.")
        self.logger.info(
            f"Predicting with {type(self.best_model_).__name__}, v{self.version}"
        )

        # Convert
        data = self.transform(data)

        # Predict
        assert self.best_model_
        predictions = self.best_model_.predict(data)

        # Convert
        if self.mode == "regression" and self.standardize:
            assert self.standardizer
            predictions = self.standardizer.reverse(predictions, column=self.target)
        elif self.mode == "classification":
            assert self.data_processor
            predictions = self.data_processor.decode_labels(predictions)

        # Stop timer
        self._prediction_time_ = (time.time() - start_time) / len(data) * 1000

        # Calculate main predictors
        self._get_main_predictors(data)

        return predictions

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Returns probabilistic prediction, only for classification.

        Parameters
        ----------
        data : pd.DataFrame
            Data to do prediction on.
        """
        start_time = time.time()
        if not self.is_fitted_:
            raise ValueError("Pipeline not yet fitted.")
        if self.mode != "classification":
            raise ValueError("Predict_proba only available for classification")
        if not hasattr(self.best_model_, "predict_proba"):
            raise ValueError(
                f"{type(self.best_model_).__name__} has no attribute predict_proba"
            )
        self.logger.info(
            f"Predicting with {type(self.best_model_).__name__}, v{self.version}"
        )

        # Convert data
        data = self.transform(data)

        # Predict
        prediction = self.best_model_.predict_proba(data)  # type: ignore -- asserted

        # Stop timer
        self._prediction_time_ = (time.time() - start_time) / len(data) * 1000

        # Calculate main predictors
        self._get_main_predictors(data)

        return prediction

    # Fit functions
    def _read_data(
        self,
        data_or_path: np.ndarray | pd.DataFrame | str | Path,
        target: list | tuple | np.ndarray | pd.Series | str | Path | None = None,
        *,
        metadata: dict[str, dict] | None = None,
    ) -> pd.DataFrame:
        """
        Read and validate data.

        Notes
        -----
        The required parameters depend on the input parameter types.

        When ``target`` is None, it is set to ``self.target`` or "target" otherwise.

        When ``data_or_path`` is path-like, then the parameters ``target`` and
        ``metadata`` are not required.
        Otherwise, when ``data_or_path`` is array-like, it either must contain a column
        name as the ``target`` parameter indicates or ``target`` must also be an
        array-like object with the same length as ``data_or_path``.

        Note: There's three combinations of data_or_path and target
        1. if data_or_path = pd.DataFrame, target = pd.Series | None | str
        2. if data_or_path = np.ndarray, target = np.ndarray | pd.Series
        3. if data_or_path = path | str, target = path | str | None

        Parameters
        ----------
        data_or_path : np.ndarray or pd.DataFrame or str or Path
            Data or path to data.
        target : np.ndarray or pd.Series or str
            Target data or column name or directory name
        *
        metadata : dict of {int : dict of {str : str or float}}, optional
            Metadata.

        Returns
        -------
        Pipeline
            The same object but with injected data.
        """
        # 1. if data_or_path = pd.DataFrame, target = ArrayLike | str | None
        if isinstance(data_or_path, pd.DataFrame):
            data = data_or_path
            # If it's a series, we check index and take the name
            if isinstance(target, pd.Series):
                if not all(data.index == target.index):
                    warn(
                        "Indices of data and target don't match. Target index will be "
                        "overwritten by data index."
                    )
                    target.index = data.index
                if target.name and self.target != target.name:
                    warn(
                        "Provided target series has a different name than initialized "
                        "target. Using series name."
                    )
                    self.target = str(target.name)
            # Then for arraylike, we check length and make sure target is not in data
            if isinstance(target, (list, tuple, pd.Series, np.ndarray)):
                if len(data) != len(target):
                    raise ValueError("Length of target and data don't match.")
                if self.target in data and (data[self.target] != target).any():
                    raise ValueError(
                        f"The column '{self.target}' column already exists in `data` "
                        f"but has different values."
                    )
                data[self.target] = target
            # If it's a string, we check its presence and update self.target
            elif isinstance(target, str):
                if target not in data:
                    raise ValueError("Provided target column not present in data.")
                self.target = target
            # If it's none, self.target is taken from __init__
            elif isinstance(target, type(None)):
                if self.target not in data:
                    raise ValueError("Initialized target column not present in data.")
            else:
                raise NotImplementedError(
                    "When data_or_path is a DataFrame, target needs to "
                    "be a Series, str or None"
                )

        # 2. if data_or_path = np.ndarray, target = ArrayLike
        elif isinstance(data_or_path, np.ndarray):
            if not isinstance(target, (np.ndarray, pd.Series, list, tuple)):
                raise NotImplementedError(
                    "If data is ndarray, target should be ArrayLike."
                )
            if len(data_or_path) != len(target):
                raise ValueError("Length of target and data don't match.")
            if isinstance(target, pd.Series):
                data = pd.DataFrame(data_or_path, index=target.index)
                if target.name:
                    self.target = str(target.name)
            else:
                data = pd.DataFrame(data_or_path)
            data[self.target] = target

        # 3. if data_or_path = path | str, target = path | str | None
        elif isinstance(data_or_path, (str, Path)):
            if isinstance(target, (str, Path)):
                self.target = str(target)
            elif not isinstance(target, type(None)):
                raise ValueError(
                    "Target must be string | Path | None when `data_or_path` is a "
                    "path-like object."
                )
            if metadata:
                warn(
                    "Parameter `metadata` is ignored when `data_or_path` is a "
                    "path-like object."
                )
            data, metadata = utils.io.merge_logs(
                parent=data_or_path, target=self.target
            )

        # 4. Error.
        else:
            raise NotImplementedError(
                "Supported data_or_path types: pd.DataFrame | np.ndarray | Path | str"
            )
        assert isinstance(data, pd.DataFrame)

        # Clean target name
        clean_target = utils.clean_feature_name(self.target)
        data = data.rename(columns={self.target: clean_target})
        self.target = clean_target

        # Finish
        self.settings["file_metadata"] = metadata or {}

        return data

    def has_new_training_data(self):
        # TODO: fix a better solution for this
        return True

    def _mode_detector(self, data: pd.DataFrame):
        """
        Detects the mode (Regression / Classification)

        parameters
        ----------
        data : pd.DataFrame
        """
        # Only run if mode is not provided
        if self.mode in ("classification", "regression"):
            return

        # Classification if string
        labels = data[self.target]
        if labels.dtype == str or labels.nunique() < 0.1 * len(data):
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

    def _set_subclasses(self):
        """
        Simple function which sets subclasses. This cannot be done
        during class initialization due to certain attributes which
        are data dependent. Data is only known at calling .fit().
        """
        assert self.mode
        self.data_processor = DataProcessor(
            target=self.target,
            drop_datetime=True,
            include_output=True,
            missing_values=self.missing_values,
            outlier_removal=self.outlier_removal,
            z_score_threshold=self.z_score_threshold,
        )
        self.data_sequencer = Sequencer(
            target=self.target,
            back=self.sequence_back,
            forward=self.sequence_forward,
            shift=self.sequence_shift,
            diff=self.sequence_diff,
        )
        self.interval_analyser = IntervalAnalyser(target=self.target)
        self.feature_processor = FeatureProcessor(
            target=self.target,
            mode=self.mode,
            is_temporal=None,
            use_wavelets=self.use_wavelets,
            extract_features=self.extract_features,
            collinear_threshold=self.information_threshold,
            verbose=self.verbose,
        )
        self.standardizer = Standardizer(target=self.target, mode=self.mode)

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
        else:
            raise NotImplementedError("Unknown Mode.")

    # Support Functions
    @staticmethod
    def sort_results_(results_: pd.DataFrame) -> pd.DataFrame:
        return results_.sort_values("worst_case", ascending=False)

    def _get_main_predictors(self, data):
        """
        Using Shapely Additive Explanations, this function calculates the main
        predictors for a given prediction and sets them into the class' memory.
        """
        # shap.TreeExplainer is not implemented for all models. So we try and fall back
        # to the feature importance given by the feature processor.
        # Note that the error would be raised when calling `TreeExplainer(best_model_)`.
        try:
            # Get shap values
            best_model_ = self.best_model_
            if type(best_model_).__module__.startswith("amplo"):
                best_model_ = best_model_.model  # type: ignore
            # Note: The error would be raised at this point.
            #  So we have not much overhead.
            shap_values = np.array(TreeExplainer(best_model_).shap_values(data))

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
            assert self.feature_processor
            fi = self.feature_processor.feature_importance_.get("rf", {})

            # Use only those columns that are present in the data
            main_predictors = {}
            missing_columns = []
            for col in data:
                if col in fi:
                    main_predictors[col] = fi[col]
                else:
                    missing_columns.append(col)

            if missing_columns:
                warn(
                    f"Some data column names are missing in the shap feature "
                    f"importance dictionary: {missing_columns}"
                )

        # Some feature names are obscure since they come from the feature processing
        # module. Here, we relate the feature importance back to the original features.
        translation = translate_features(list(main_predictors))
        scores = {}
        for key, features in translation.items():
            for feat in features:
                scores[feat] = scores.get(feat, 0.0) + main_predictors[key]
        # Normalize
        total_score = np.sum(list(scores.values()))
        for key in scores:
            scores[key] /= total_score

        # Set attribute
        self.main_predictors_ = scores

        # Add to settings: [{"feature": "feature_name", "score": 1}, ...]
        scores_df = pd.DataFrame({"feature": scores.keys(), "score": scores.values()})
        scores_df.sort_values("score", ascending=False, inplace=True)
        self.settings["main_predictors"] = scores_df.to_dict("records")

        return scores
