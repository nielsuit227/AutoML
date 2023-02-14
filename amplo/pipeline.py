#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd
from shap import TreeExplainer
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import KFold, StratifiedKFold

from amplo.automl.data_processing import DataProcessor
from amplo.automl.feature_processing.feature_processing import (
    FeatureProcessor,
    get_required_columns,
    translate_features,
)
from amplo.automl.grid_search import OptunaGridSearch
from amplo.automl.modelling import Modeller, get_model
from amplo.automl.standardization import Standardizer
from amplo.base import AmploObject, BaseEstimator
from amplo.base.objects import LoggingMixin, Result
from amplo.observation import DataObserver, ModelObserver
from amplo.utils import clean_feature_name, io, logging
from amplo.validation import ModelValidator

__all__ = ["Pipeline"]

warnings.filterwarnings("ignore", message="lbfgs failed to converge")
pd.options.mode.copy_on_write = True


class Pipeline(AmploObject, LoggingMixin):
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
        # Modelling
        standardize: bool = False,
        cv_shuffle: bool = True,
        cv_splits: int = 10,
        store_models: bool = False,
        # Grid search
        grid_search_timeout: int = 3600,
        n_grid_searches: int = 2,
        n_trials_per_grid_search: int = 250,
        # Other
        **kwargs,
    ):
        AmploObject.__init__(self)

        # Initialize Logger
        LoggingMixin.__init__(self, verbose=verbose)
        if logging_path is None:
            logging_path = f"{Path(main_dir)}/AutoML.log"
        if logging_to_file:
            logging.add_file_handler(logging_path)

        # Input checks: validity
        if mode not in (None, "regression", "classification"):
            raise ValueError("Supported models: {'regression', 'classification', None}")
        if not 0 < information_threshold < 1:
            raise ValueError("Information threshold must be within (0, 1) interval.")

        # Input checks: advices
        if kwargs:
            warn(f"Got unexpected keyword arguments that are not handled: {kwargs}")

        # Main settings
        self.metadata: dict[str, dict[str, Any]] = {}
        self.main_dir = f"{Path(main_dir)}/"  # assert '/' afterwards
        self.target = target
        self.name = name
        self.version = version
        self.mode = mode or ""
        self.objective = objective or ""
        self.logging_to_file = logging_to_file
        self.logging_path = logging_path
        self.verbose = verbose

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

        # Modelling
        self.standardize = standardize
        self.cv_shuffle = cv_shuffle
        self.cv_splits = cv_splits
        self.store_models = store_models

        # Grid search
        self.grid_search_timeout = grid_search_timeout
        self.n_grid_searches = n_grid_searches
        self.n_trials_per_grid_search = n_trials_per_grid_search

        # Set version
        self.version = version if version else 1

        # Objective & Scorer
        if self.objective and self.objective not in get_scorer_names():
            raise ValueError(f"Invalid objective.\nPick from {get_scorer_names()}")

        # Required sub-classes
        self.data_processor: DataProcessor
        self.feature_processor: FeatureProcessor
        self.standardizer: Standardizer
        self.data_observations: list[dict[str, str | bool]] = []
        self.model_observations: list[dict[str, str | bool]] = []

        # Instance initiating
        self.best_model_: BaseEstimator
        self.results_: list[Result] = []
        self.is_fitted_ = False
        self.validation: dict[str, Any] = {}

        # Monitoring
        self.file_delta_: dict[str, list[str]]
        self._prediction_time_: float
        self.main_predictors_: dict[str, float]

    def fit(
        self,
        data: npt.NDArray[Any] | pd.DataFrame | str | Path,
        target: npt.NDArray[Any] | pd.Series | str | None = None,
        *,
        metadata: dict[str, dict[str, Any]] | None = None,
        model: str | None = None,
        feature_set: str | None = None,
    ):
        """
        Fit the full AutoML pipeline.
            1. Prepare data for training
            2. Train / optimize models
            3. Prepare Production Files
                Nicely organises all required scripts / files to make a prediction

        Parameters
        ----------
        data_or_path : npt.NDArray[Any] or pd.DataFrame or str or Path
            Data or path to data. Propagated to `self.data_preparation`.
        target : npt.NDArray[Any] or pd.Series or str
            Target data or column name. Propagated to `self.data_preparation`.
        *
        metadata : dict of {int : dict of {str : str or float}}, optional
            Metadata. Propagated to `self.data_preparation`.
        model : str, optional
            Limits model training and grid search to a specific model.
        feature_set : str, optional
            Limits model training and grid search to a specific feature set.
        params : dict, optional
            Constrain parameters for fitting conclusion.
            Propagated to `self.conclude_fitting`.
        """
        # Starting
        self.logger.info(f"\n\n*** Starting Amplo AutoML - {self.name} ***\n\n")

        # Reading data
        data = self._read_data(data, target, metadata=metadata)

        # Detect mode (classification / regression)
        self._mode_detector(data)

        self._set_subclasses()

        # Preprocess Data
        data = self.data_processor.fit_transform(data)

        # Extract and select features
        data = self.feature_processor.fit_transform(data, feature_set=feature_set)

        # Standardize
        if self.standardize:
            data = self.standardizer.fit_transform(data)

        # Model Training
        ################
        for feature_set_, cols in self.feature_processor.feature_sets_.items():
            if feature_set and feature_set_ != feature_set:
                continue

            self.logger.info(f"Fitting modeller on: {feature_set_}")
            feature_data: pd.DataFrame = data[cols + [self.target]]
            results_ = Modeller(
                target=self.target,
                mode=self.mode,
                cv=self.cv,
                objective=self.objective,
                verbose=self.verbose,
                feature_set=feature_set_,
                model=model,
            ).fit(feature_data)
            self.results_.extend(results_)
        self.sort_results()

        # Optimize Hyper parameters
        for model_, feature_set in self.grid_search_iterations():
            if feature_set not in self.feature_processor.feature_sets_:
                raise ValueError(f"Found invalid feature set: '{feature_set}'")
            self.logger.info(
                f"Starting Hyper Parameter Optimization for {model_} on "
                f"{feature_set} features ({len(data)} samples, "
                f"{len(self.feature_processor.feature_sets_[feature_set])} features)"
            )
            results_ = OptunaGridSearch(
                get_model(model_),
                target=self.target,
                timeout=self.grid_search_timeout,
                feature_set=feature_set,
                cv=self.cv,
                n_trials=self.n_trials_per_grid_search,
                scoring=self.objective,
                verbose=self.verbose,
            ).fit(data)
            self.results_.extend(results_)
        self.sort_results()

        self.train_val_best(data)

        self.data_observations = DataObserver().observe(
            data, self.mode, self.target, self.data_processor.dummies_
        )
        self.model_observations = ModelObserver().observe(
            self.best_model_, data, self.target, self.mode
        )

        self.is_fitted_ = True
        self.logger.info("All done :)")
        logging.del_file_handlers()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("Pipeline not yet fitted.")

        # Process data
        data = self.data_processor.transform(data)

        # Convert Features
        data = self.feature_processor.transform(
            data, feature_set=self.best_feature_set_
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
        predictions = self.best_model_.predict(data)

        # Convert
        if self.mode == "regression" and self.standardize:
            predictions = self.standardizer.reverse(predictions, column=self.target)
        elif self.mode == "classification":
            predictions = self.data_processor.decode_labels(predictions)

        # Stop timer
        self._prediction_time_ = (time.time() - start_time) / len(data) * 1000

        # Calculate main predictors
        self._get_main_predictors(data)

        return predictions

    def predict_proba(self, data: pd.DataFrame) -> npt.NDArray[Any]:
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
        prediction = self.best_model_.predict_proba(data)

        # Stop timer
        self._prediction_time_ = (time.time() - start_time) / len(data) * 1000

        # Calculate main predictors
        self._get_main_predictors(data)

        return prediction

    # Support functions
    def grid_search_iterations(self) -> list[tuple[str, str]]:
        """Takes top `n_grid_searches` models / feature set combi's from results"""
        return [
            (self.results_[i].model, self.results_[i].feature_set)
            for i in range(self.n_grid_searches)
        ]

    def train_val_best(self, data: pd.DataFrame):
        """Arranges settings and parameter file."""

        # Train model on all training data
        best_model_ = get_model(self.best_model_str_)
        best_model_.set_params(**self.best_params_)
        best_model_.fit(data[self.best_features_], data[self.target])
        self.best_model_ = best_model_

        # Prune Data Processor
        required_features = get_required_columns(
            self.feature_processor.feature_sets_[self.best_feature_set_]
        )
        self.data_processor.prune_features(required_features)

        # Set feature set
        self.feature_processor.set_feature_set(self.best_feature_set_)

        # Validation
        self.validation = ModelValidator(
            target=self.target,
            cv=self.cv,
            verbose=self.verbose,
        ).validate(model=best_model_, data=data, mode=self.mode)

    def _read_data(
        self,
        data_or_path: npt.NDArray[Any] | pd.DataFrame | str | Path,
        target: list[Any]
        | tuple[Any]
        | npt.NDArray[Any]
        | pd.Series
        | str
        | Path
        | None = None,
        *,
        metadata: dict[str, dict[str, Any]] | None = None,
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
        2. if data_or_path = npt.NDArray[Any], target = npt.NDArray[Any] | pd.Series
        3. if data_or_path = path | str, target = path | str | None

        Parameters
        ----------
        data_or_path : npt.NDArray[Any] or pd.DataFrame or str or Path
            Data or path to data.
        target : npt.NDArray[Any] or pd.Series or str
            Target data or column name or directory name
        *
        metadata : dict of {int : dict of {str : str or float}}, optional
            Metadata.

        Returns
        -------
        Pipeline
            The same object but with injected data.
        """
        self.logger.info("Reading data.")
        # 1. if data_or_path = pd.DataFrame, target = ArrayLike | str | None
        if isinstance(data_or_path, pd.DataFrame):
            self.logger.debug("Detected pandas dataframe. Checking target.")
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
            self.logger.debug("Detected numpy array. Checking target.")
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
            self.logger.debug("Detected path. ")
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
            data, metadata = io.merge_logs(parent=data_or_path, target=self.target)

        # 4. Error.
        else:
            raise NotImplementedError(
                "Supported data_or_path types: pd.DataFrame | np.ndarray | Path | str"
            )

        # Safety check
        assert isinstance(data, pd.DataFrame)

        # Clean target name
        clean_target = clean_feature_name(self.target)
        data = data.rename(columns={self.target: clean_target})
        self.target = clean_target

        # Finish
        self.metadata = metadata or {}
        self.logger.info(
            f"Data contains {len(data)} samples and {len(data.keys())} columns."
        )

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
        self.logger.debug("Detecting mode.")

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
        self.logger.debug("Setting subclasses.")

        self.data_processor = DataProcessor(
            target=self.target,
            drop_datetime=True,
            include_output=True,
            missing_values=self.missing_values,
            outlier_removal=self.outlier_removal,
            z_score_threshold=self.z_score_threshold,
            verbose=self.verbose,
        )
        self.feature_processor = FeatureProcessor(
            target=self.target,
            mode=self.mode,
            is_temporal=None,
            use_wavelets=self.use_wavelets,
            extract_features=self.extract_features,
            collinear_threshold=self.information_threshold,
            verbose=self.verbose,
        )
        self.standardizer = Standardizer(
            target=self.target, mode=self.mode, verbose=self.verbose
        )

    # Support Functions
    def sort_results(self) -> list[Result]:
        self.results_.sort(reverse=True)
        return self.results_

    def _get_main_predictors(self, data: pd.DataFrame) -> dict[str, float]:
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
            if best_model_ is not None and hasattr(best_model_, "model"):
                best_model_ = best_model_.model
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

        except Exception:
            # Get shap feature importance
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
        self.main_predictors_ = {}
        for key, features in translation.items():
            for feat in features:
                self.main_predictors_[feat] = (
                    self.main_predictors_.get(feat, 0.0) + main_predictors[key]
                )
        # Normalize
        total_score = np.sum(list(self.main_predictors_.values()))
        for key in self.main_predictors_:
            self.main_predictors_[key] /= total_score

        return self.main_predictors_

    # Properties
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

    @property
    def best_feature_set_(self) -> str:
        if not self.results_:
            raise ValueError("No results available.")
        return self.results_[0].feature_set

    @property
    def best_features_(self) -> list[str]:
        feature_sets = self.feature_processor.feature_selector.feature_sets_
        return feature_sets[self.best_feature_set_]

    @property
    def best_model_str_(self) -> str:
        if not self.results_:
            raise ValueError("No results available.")
        return self.results_[0].model

    @property
    def best_params_(self) -> dict[str, Any]:
        if not self.results_:
            raise ValueError("No results available.")
        return io.parse_json(self.results_[0].params)  # type: ignore[return-value]

    @property
    def best_score_(self) -> float:
        if not self.results_:
            raise ValueError("No results available.")
        return self.results_[0].worst_case
