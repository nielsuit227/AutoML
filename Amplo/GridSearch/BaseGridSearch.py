import time
import copy
import random
import warnings
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
from sklearn.metrics import SCORERS


# noinspection PyUnresolvedReferences
class BaseGridSearch:

    def __init__(self, model, params=None, cv=None, scoring='accuracy',
                 timeout=3600, candidates=250, verbose=0):
        """
        Basic exhaustive grid search.
        @param model: Model object to optimize
        @param cv: Cross-Validator for scoring
        @param scoring: make_scorer for scoring
        """
        # Args
        self.model = model
        self.params = params
        self.cv = cv
        self.scoring = SCORERS[scoring] if isinstance(scoring, str) else scoring
        self.timeout = timeout
        self.nTrials = candidates
        self.verbose = verbose
        self.mode = 'regression' if 'Regressor' in type(model).__name__ else 'classification'
        self.binary = True
        self.samples = None

        # Checks
        assert model is not None, 'Model not provided'
        assert self.scoring in SCORERS.values(), 'Chose a scorer from sklearn.metrics.SCORERS'
        
        # Initiate
        self.parsedParams = []
        self.result = []

    def _get_hyper_params(self):
        # Parameters for both Regression / Classification
        if type(self.model).__name__ == 'LinearRegression':
            return {}
        elif type(self.model).__name__ == 'Lasso' or \
                'Ridge' in type(self.model).__name__:
            return {
                'alpha': np.logspace(-5, 0, 25).tolist(),
            }
        elif 'SV' in type(self.model).__name__:
            return {
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 0.5, 1],
                'C': np.logspace(-5, 0, 25).tolist(),
            }
        elif 'KNeighbors' in type(self.model).__name__:
            return {
                'n_neighbors': np.linspace(5, min(50, int(self.samples / 10)), 5).astype('int').tolist(),
                'weights': ['uniform', 'distance'],
                'leaf_size': np.linspace(1, min(150, int(self.samples / 10), 5).astype('int').tolist()),
                'n_jobs': [mp.cpu_count() - 1],
            }
        elif 'MLP' in type(self.model).__name__:
            return {
                'hidden_layer_sizes': [(100,), (100, 100), (100, 50), (200, 200), (200, 100), (200, 50),
                                       (50, 50, 50, 50)],
                'learning_rate': ['adaptive', 'invscaling'],
                'alpha': [1e-4, 1e-3, 1e-5],
                'shuffle': [False],
            }

        # Regressor specific hyper parameters
        elif self.mode == 'regression':
            if type(self.model).__name__ == 'SGDRegressor':
                return {
                    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': np.logspace(-5, 0, 5).tolist(),
                }
            elif type(self.model).__name__ == 'DecisionTreeRegressor':
                return {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': np.linspace(3, min(15, int(np.log2(self.samples))), 4).astype('int').tolist(),
                }
            elif type(self.model).__name__ == 'AdaBoostRegressor':
                return {
                    'n_estimators': [25, 50, 75, 100, 150, 200, 250],
                    'loss': ['linear', 'square', 'exponential'],
                    'learning_rate': np.logspace(-5, 0, 10).tolist()
                }
            elif type(self.model).__name__ == 'BaggingRegressor':
                return {
                    # 'n_estimators': [5, 10, 15, 25, 50],
                    'max_features': [1, 0.9, 0.8, 0.5],
                    'max_samples': [1, 0.9, 0.8, 0.5],
                    'bootstrap': [False, True],
                    'bootstrap_features': [True, False],
                    'n_jobs': [mp.cpu_count() - 1],
                }
            elif type(self.model).__name__ == 'GradientBoostingRegressor':
                return {
                    'loss': ['ls', 'lad', 'huber'],
                    'learning_rate': np.logspace(-5, 0, 10).tolist(),
                    'max_depth': np.linspace(3, min(15, int(np.log2(self.samples))), 4).astype('int').tolist(),
                }
            elif type(self.model).__name__ == 'HistGradientBoostingRegressor':
                return {
                    'max_iter': [100, 250],
                    'max_bins': [100, 150, 200, 255],
                    'loss': ['least_squares', 'least_absolute_deviation'],
                    'l2_regularization': np.logspace(-5, 0, 5).tolist(),
                    'learning_rate': np.logspace(-5, 0, 5).tolist(),
                    'max_leaf_nodes': [30, 50, 100, 150, 200, 250],
                    'early_stopping': [True],
                }
            elif type(self.model).__name__ == 'RandomForestRegressor':
                return {
                    'criterion': ['squared_error', 'absolute_error'],
                    'max_depth': np.linspace(3, min(15, int(np.log2(self.samples))), 4).astype('int').tolist(),
                    'max_features': ['auto', 'sqrt'],
                    'min_samples_split': [2, 25, 50],
                    'min_samples_leaf': np.logspace(0, min(3, int(np.log10(100)) - 1), 5).astype('int').tolist(),
                    'bootstrap': [True, False],
                }
            elif type(self.model).__name__ == 'CatBoostRegressor':
                return {
                    'verbose': [0],
                    'loss_function': ['MAE', 'RMSE'],
                    'allow_writing_files': [False],
                    'early_stopping_rounds': [100],
                    'learning_rate': np.logspace(-5, 0, 5).tolist(),
                    'l2_leaf_reg': np.logspace(-5, 0, 5).tolist(),
                    'depth': [3, 5, 7, 10],
                    'min_data_in_leaf': np.logspace(0, min(3, int(np.log10(100)) - 1), 5).astype('int').tolist(),
                    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
                }
            elif type(self.model).__name__ == 'XGBRegressor':
                return {
                    'max_depth': np.linspace(3, min(15, int(np.log2(self.samples))), 4).astype('int').tolist(),
                    'booster': ['gbtree', 'gblinear', 'dart'],
                    'learning_rate': np.logspace(-5, 0, 10).tolist(),
                    'verbosity': [0],
                    'n_jobs': [mp.cpu_count() - 1],
                }
            elif type(self.model).__name__ == 'LGBMRegressor':
                return {
                    'verbosity': [-1],
                    'num_leaves': [10, 50, 100, 150],
                    'min_child_samples': [1, 50, 500, 1000],
                    'min_child_weight': np.logspace(-5, 0, 5).tolist(),
                    'subsample': [1, 0.9, 0.8, 0.5],
                    'colsample_bytree': [1, 0.9, 0.8, 0.5],
                    'reg_alpha': np.logspace(-5, 0, 5).tolist(),
                    'reg_lambda': np.logspace(-5, 0, 5).tolist(),
                    'n_jobs': [mp.cpu_count() - 1],
                }

        # Classification specific hyper parameters
        elif self.mode == 'classification':
            if type(self.model).__name__ == 'SGDClassifier':
                return {
                    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': np.logspace(-5, 0, 5).tolist(),
                    'max_iter': [250, 1000, 1500],
                }
            elif type(self.model).__name__ == 'DecisionTreeClassifier':
                return {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': np.linspace(3, min(50, int(np.log2(self.samples))), 4).astype('int').tolist(),
                }
            elif type(self.model).__name__ == 'AdaBoostClassifier':
                return {
                    'n_estimators': [25, 50, 100, 150, 250],
                    'learning_rate': np.logspace(-5, 0, 10).tolist()
                }
            elif type(self.model).__name__ == 'BaggingClassifier':
                return {
                    # 'n_estimators': [5, 10, 15, 25, 50],
                    'max_features': [1, 0.9, 0.8, 0.5],
                    'max_samples': [1, 0.9, 0.8, 0.5],
                    'bootstrap': [False, True],
                    'bootstrap_features': [True, False],
                    'n_jobs': [mp.cpu_count() - 1],
                }
            elif type(self.model).__name__ == 'GradientBoostingClassifier':
                return {
                    'loss': ['deviance', 'exponential'],
                    'learning_rate': np.logspace(-5, 0, 10).tolist(),
                    'max_depth': np.linspace(3, min(15, int(np.log2(self.samples))), 4).astype('int').tolist(),
                }
            elif type(self.model).__name__ == 'HistGradientBoostingClassifier':
                return {
                    'max_iter': [100, 250, 500, 1000],
                    'max_bins': [100, 175, 255],
                    'l2_regularization': np.logspace(-5, 0, 5).tolist(),
                    'learning_rate': np.logspace(-5, 0, 5).tolist(),
                    'max_leaf_nodes': [30, 100, 200],
                    'early_stopping': [True]
                }
            elif type(self.model).__name__ == 'RandomForestClassifier':
                return {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': np.linspace(3, min(15, int(np.log2(self.samples))), 4).astype('int').tolist(),
                    'max_features': ['auto', 'sqrt'],
                    'min_samples_split': [2, 50, 500],
                    'min_samples_leaf': np.logspace(0, min(3, int(np.log10(100)) - 1), 5).astype('int').tolist(),
                    'bootstrap': [True, False],
                }
            elif type(self.model).__name__ == 'CatBoostClassifier':
                return {
                    'verbose': [0],
                    'loss_function': ['Logloss'] if self.binary else ['MultiClass'],
                    'allow_writing_files': [False],
                    'early_stopping_rounds': [100],
                    'learning_rate': np.logspace(-5, 0, 5).tolist(),
                    'l2_leaf_reg': np.logspace(-5, 0, 5).tolist(),
                    'depth': [3, 5, 7, 10],
                    'min_data_in_leaf': np.logspace(0, min(3, int(np.log10(100)) - 1), 5).astype('int').tolist(),
                    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
                }
            elif type(self.model).__name__ == 'XGBClassifier':
                return {
                    "objective": ["binary:logistic"] if self.binary else ['multi:softprob'],
                    'max_depth': np.linspace(3, min(15, int(np.log2(self.samples))), 4).astype('int').tolist(),
                    'booster': ['gbtree', 'gblinear', 'dart'],
                    'learning_rate': np.logspace(-5, 0, 10).tolist(),
                    'verbosity': [0],
                    'n_jobs': [mp.cpu_count() - 1],
                    'scale_pos_weight': [0, 10, 50, 100],
                }
            elif type(self.model).__name__ == 'LGBMClassifier':
                return {
                    "objective": ["binary"] if self.binary else ['multiclass'],
                    'verbosity': [-1],
                    'num_leaves': [10, 50, 100, 150],
                    'min_child_samples': [1, 50, 500, 1000],
                    'min_child_weight': np.logspace(-5, 0, 5).tolist(),
                    'subsample': np.logspace(-5, 0, 5).tolist(),
                    'colsample_bytree': np.logspace(-5, 0, 5).tolist(),
                    'reg_alpha': np.logspace(-5, 0, 5).tolist(),
                    'reg_lambda': np.logspace(-5, 0, 5).tolist(),
                    'n_jobs': [mp.cpu_count() - 1],
                }

        # Raise error if nothing is returned
        warnings.warn('Hyper parameter tuning not implemented for {}'.format(type(self.model).__name__))
        return {}

    def _parse_params(self):
        if len(self.params.items()) > 0:
            k, v = zip(*self.params.items())
            self.parsedParams = [dict(zip(k, v)) for v in itertools.product(*self.params.values())]
            random.shuffle(self.parsedParams)
            print('[GridSearch] Optimizing {}, {}-folds with {} parameter combinations, {} runs.' .format(
                type(self.model).__name__, self.cv.n_splits, len(self.parsedParams),
                len(self.parsedParams) * self.cv.n_splits))
        else:
            self.parsedParams = [{}]

    def fit(self, x, y):
        start_time = time.time()
        # todo check data
        # Parse Params
        if self.mode == 'classification':
            if len(np.unique(y)) == 2:
                self.binary = True
            else:
                self.binary = False
        # Convert to Numpy
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = np.array(x)
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = np.array(y).reshape((-1))
        self.samples = len(y)

        # Get params
        if self.params is None:
            self.params = self._get_hyper_params()
        self._parse_params()

        # Loop through parameters
        for i, param in tqdm(enumerate(self.parsedParams)):
            # Timeout or trials limit
            if time.time() - start_time > self.timeout or i > self.nTrials:
                return pd.DataFrame(self.result)

            # Scoring and timing vector
            scoring = []
            timing = []

            # Cross Validation
            for train_ind, val_ind in self.cv.split(x, y):
                # Start Timer
                t = time.time()

                # Split data
                x_train, x_val = x[train_ind], x[val_ind]
                y_train, y_val = y[train_ind], y[val_ind]

                # Model training
                model = copy.deepcopy(self.model)
                model.set_params(**param)
                model.fit(x_train, y_train)

                # Results
                scoring.append(self.scoring(model, x_val, y_val))
                timing.append(time.time() - t)

            # Output Printing
            if self.verbose > 0:
                print('[GridSearch][{}] Score: {:.4f} \u00B1 {:.4f} (in {:.1f} seconds) (Best score so-far: {:.4f}'
                      ' \u00B1 {:.4f}) ({} / {})'.format(datetime.now().strftime('%H:%M'), np.mean(scoring),
                                                         np.std(scoring), np.mean(timing), self.best[0], self.best[1],
                                                         i + 1, len(self.parsedParams)))

            self.result.append({
                'date': datetime.today().strftime('%d %b %y'),
                'model': type(model).__name__,
                'mean_objective': np.mean(scoring),
                'std_objective': np.std(scoring),
                'worst_case': np.mean(scoring) - np.std(scoring),
                'params': param,
                'mean_time': np.mean(timing),
                'std_time': np.std(timing),
            })
        return pd.DataFrame(self.result)
