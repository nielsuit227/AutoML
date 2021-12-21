import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from scipy.stats import uniform
from scipy.stats import randint
from scipy.stats import loguniform
from sklearn.metrics import SCORERS
from sklearn.model_selection import KFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV


class HalvingGridSearch:

    def __init__(self, model, params=None, cv=KFold(n_splits=10), scoring='accuracy',
                 candidates=250, verbose=0):
        # Params
        self.model = model
        self.params = params
        self.cv = cv
        self.scoring = scoring
        self.candidates = candidates
        self.verbose = verbose
        self.samples = None
        self.binary = False

        # Halving settings
        self.resource = 'n_samples'
        self.max_resource = 'auto'
        self.min_resource = 200
        self.set_resources()

        # Tests
        assert model is not None, 'Model not provided'
        assert scoring in SCORERS.keys(), 'Scoring not recognized, pick from sklearn.metrics.SCORERS'
        if params is not None:
            assert isinstance(params, dict), 'Provided params need to be of type dict'
            assert all([hasattr(x, 'rvs') for x in params.values()]), 'Param values should have rvs'

    def set_resources(self):
        if 'CatBoost' in type(self.model).__name__:
            self.resource = 'n_estimators'
            self.max_resource = 3000
            self.min_resource = 250
        if self.model.__module__ == 'sklearn.ensemble._bagging' or \
                self.model.__module__ == 'xgboost.sklearn' or \
                self.model.__module__ == 'lightgbm.sklearn' or \
                self.model.__module__ == 'sklearn.ensemble._forest':
            self.resource = 'n_estimators'
            self.max_resource = 1500
            self.min_resource = 50

    def _get_hyper_params(self):
        # Parameters for both Regression / Classification
        if type(self.model).__name__ == 'LinearRegression':
            return {}
        elif type(self.model).__name__ == 'Lasso' or \
                'Ridge' in type(self.model).__name__:
            return {
                'alpha': uniform(0, 10),
            }
        elif 'SV' in type(self.model).__name__:
            return {
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 0.5, 1],
                'C': uniform(0, 10),
            }
        elif 'KNeighbors' in type(self.model).__name__:
            return {
                'n_neighbors': randint(5, 50),
                'weights': ['uniform', 'distance'],
                'leaf_size': randint(10, 150),
                'n_jobs': [mp.cpu_count() - 1],
            }

        # Regressor specific hyper parameters
        elif type(self.model).__name__ == 'SGDRegressor':
            return {
                'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': randint(0, 5),
            }
        elif type(self.model).__name__ == 'DecisionTreeRegressor':
            return {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'max_depth': randint(5, 50),
            }
        elif type(self.model).__name__ == 'AdaBoostRegressor':
            return {
                'n_estimators': randint(25, 250),
                'loss': ['linear', 'square', 'exponential'],
                'learning_rate': uniform(0, 1)
            }
        elif type(self.model).__name__ == 'GradientBoostingRegressor':
            return {
                'loss': ['ls', 'lad', 'huber'],
                'learning_rate': uniform(0, 1),
                'max_depth': randint(3, min(10, int(np.log2(self.samples)))),
                'n_estimators': randint(100, 500),
                'min_samples_leaf': randint(1, min(1000, int(self.samples / 10))),
                'max_features': uniform(0, 1),
                'subsample': uniform(0, 1),
            }
        elif type(self.model).__name__ == 'HistGradientBoostingRegressor':
            return {
                'loss': ['least_squares', 'least_absolute_deviation'],
                'learning_rate': loguniform(0.001, 0.5),
                'max_iter': randint(100, 250),
                'max_leaf_nodes': randint(30, 150),
                'max_depth': randint(3, min(10, int(np.log2(self.samples)))),
                'max_bins': randint(100, 255),
                'l2_regularization': uniform(0, 10),
                'early_stopping': [True],
            }
        elif type(self.model).__name__ == 'RandomForestRegressor':
            return {
                'criterion': ['squared_error', 'absolute_error'],
                'max_depth': randint(3, min(15, int(np.log2(self.samples)))),
                'max_features': ['auto', 'sqrt'],
                'min_samples_split': randint(2, 50),
                'min_samples_leaf': randint(1, min(1000, int(self.samples / 10))),
                'bootstrap': [True, False],
            }
        elif type(self.model).__name__ == 'CatBoostRegressor':
            return {
                'verbose': [0],
                'early_stopping_rounds': [100],
                'od_pval': [1e-5],
                'loss_function': ['MAE', 'RMSE'],
                'learning_rate': loguniform(0.001, 0.5),
                'l2_leaf_reg': uniform(0, 10),
                'depth': randint(3, min(10, int(np.log2(self.samples)))),
                'min_data_in_leaf': randint(1, min(1000, int(self.samples / 10))),
                'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
            }
        elif type(self.model).__name__ == 'XGBRegressor':
            return {
                'max_depth': randint(3, 15),
                'booster': ['gbtree', 'gblinear', 'dart'],
                'learning_rate': loguniform(0.001, 0.5),
            }
        elif type(self.model).__name__ == 'LGBMRegressor':
            return {
                    'num_leaves': randint(10, 150),
                    'min_child_samples': randint(1, min(1000, int(self.samples / 10))),
                    'min_child_weight': uniform(0, 1),
                    'subsample': uniform(0, 1),
                    'colsample_bytree': uniform(0, 1),
                    'reg_alpha': uniform(0, 1),
                    'reg_lambda': uniform(0, 1),
                }

        # Classification specific hyper parameters
        elif type(self.model).__name__ == 'SGDClassifier':
            return {
                'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': uniform(0, 10),
                'max_iter': randint(250, 1000),
            }
        elif type(self.model).__name__ == 'DecisionTreeClassifier':
            return {
                'criterion': ['gini', 'entropy'],
                'max_depth': randint(5, min(50, int(np.log2(self.samples)))),
            }
        elif type(self.model).__name__ == 'AdaBoostClassifier':
            return {
                'n_estimators': randint(25, 250),
                'learning_rate': loguniform(0.001, 0.5)
            }
        elif type(self.model).__name__ == 'BaggingClassifier':
            return {
                # 'n_estimators': [5, 10, 15, 25, 50],
                'max_features': uniform(0, 1),
                'bootstrap': [False, True],
                'bootstrap_features': [True, False],
                'n_jobs': [mp.cpu_count() - 1],
            }
        elif type(self.model).__name__ == 'GradientBoostingClassifier':
            return {
                'loss': ['deviance', 'exponential'],
                'learning_rate': loguniform(0.001, 0.5),
                'max_depth': randint(3, min(15, int(np.log2(self.samples)))),
                'n_estimators': randint(100, 500),
                'min_samples_leaf': randint(1, min(1000, int(self.samples / 10))),
                'max_features': uniform(0, 1),
                'subsample': uniform(0, 1),
            }
        elif type(self.model).__name__ == 'HistGradientBoostingClassifier':
            return {
                'learning_rate': loguniform(0.001, 0.5),
                'max_iter': randint(100, 250),
                'max_leaf_nodes': randint(30, 150),
                'max_depth': randint(3, min(10, int(np.log2(self.samples)))),
                'min_samples_leaf': randint(1, min(1000, int(self.samples / 10))),
                'l2_regularization': loguniform(0.0001, 10),
                'max_bins': randint(100, 255),
                'early_stopping': [True]
            }
        elif type(self.model).__name__ == 'RandomForestClassifier':
            return {
                'criterion': ['gini', 'entropy'],
                'max_depth': randint(3, min(15, int(np.log2(self.samples)))),
                'max_features': ['auto', 'sqrt'],
                'min_samples_split': randint(2, 50),
                'min_samples_leaf': randint(1, min(1000, int(self.samples / 10))),
                'bootstrap': [True, False],
            }
        elif type(self.model).__name__ == 'CatBoostClassifier':
            return {
                "verbose": [0],
                "early_stopping_rounds": [100],
                "od_pval": [1e-5],
                'loss_function': ['Logloss' if self.binary else 'MultiClass'],
                'eval_metric': ['Logloss'],
                'learning_rate': loguniform(0.001, 0.5),
                'l2_leaf_reg': loguniform(0.0001, 10),
                'depth': randint(1,  min(10, int(np.log2(self.samples)))),
                'min_data_in_leaf': randint(1, min(1000, int(self.samples / 10))),
                'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
            }
        elif type(self.model).__name__ == 'XGBClassifier':
            return {
                "objective": ["binary:logistic"] if self.binary else ['multi:softprob'],
                'max_depth': randint(3,  min(10, int(np.log2(self.samples)))),
                'booster': ['gbtree', 'gblinear', 'dart'],
                'learning_rate': loguniform(0.001, 0.5),
                'verbosity': [0],
                'scale_pos_weight': uniform(0, 100),
                "alpha": loguniform(0.0001, 1)
            }
        elif type(self.model).__name__ == 'LGBMClassifier':
            return {
                    "objective": "binary" if self.binary else 'multiclass',
                    "verbosity": [-1],
                    "boosting_type": ["gbdt"],
                    "lambda_l1": loguniform(1e-3, 10),
                    "lambda_l2": loguniform(1e-3, 10),
                    'num_leaves': randint(2, 256),
                    'min_child_samples': randint(1, min(1000, int(self.samples / 10))),
                    'min_child_weight': loguniform(1e-5, 1),
                    'subsample': uniform(0, 1),
                    'colsample_bytree': uniform(0, 1),
        }

        else:
            # Raise error if nothing is returned
            warnings.warn('Hyper parameter tuning not implemented for {}'.format(type(self.model).__name__))
            return {}

    def fit(self, x, y):
        # Update minimum resource for samples (based on dataset)
        if self.resource == 'n_samples':
            self.min_resource = int(0.2 * len(x)) if len(x) > 5000 else len(x)

        # Parameters
        if len(np.unique(y)) == 2:
            self.binary = True
        self.samples = len(y)
        if self.params is None:
            self.params = self._get_hyper_params()

        # Set up and run grid search
        halving_random_search = HalvingRandomSearchCV(self.model,
                                                      self.params,
                                                      n_candidates=self.candidates,
                                                      resource=self.resource,
                                                      max_resources=self.max_resource,
                                                      min_resources=self.min_resource,
                                                      cv=self.cv,
                                                      scoring=self.scoring,
                                                      factor=3,
                                                      n_jobs=mp.cpu_count() - 1,
                                                      verbose=self.verbose)
        halving_random_search.fit(x, y)

        # Parse results
        scikit_results = pd.DataFrame(halving_random_search.cv_results_)
        results = pd.DataFrame({
            'date': datetime.today().strftime('%d %b %y'),
            'model': type(self.model).__name__,
            'params': scikit_results['params'],
            'mean_objective': scikit_results['mean_test_score'],
            'std_objective': scikit_results['std_test_score'],
            'worst_case': scikit_results['mean_test_score'] - scikit_results['std_test_score'],
            'mean_time': scikit_results['mean_fit_time'],
            'std_time': scikit_results['std_fit_time'],
        })

        # Update resource in results
        if self.resource != 'n_samples':
            for i in range(len(results)):
                results.loc[results.index[i], 'params'][self.resource] = self.max_resource

        return results
