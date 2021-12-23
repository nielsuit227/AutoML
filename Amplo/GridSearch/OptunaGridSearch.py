import copy
import time
import optuna
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import SCORERS


class OptunaGridSearch:

    def __init__(self, model, cv=KFold(n_splits=10), scoring='accuracy', verbose=0, timeout=3600,
                 candidates=250):
        """
        Wrapper for Optuna Grid Search. Takes any model support by Amplo.AutoML.Modelling.
        The parameter search space is predefined for each model.

        Parameters
        ----------
        model obj: Model object to optimize
        cv obj: Scikit CV object
        scoring str: From Scikits Scorers*
        verbose int: How much to print
        timeout int: Time limit of optimization
        candidates int: Candidate limits to evaluate
        """
        self.model = model
        if hasattr(model, 'is_fitted'):
            assert not model.is_fitted(), 'Model already fitted'
        self.cv = cv
        self.scoring = SCORERS[scoring] if isinstance(scoring, str) else scoring
        self.verbose = verbose
        self.timeout = timeout
        self.nTrials = candidates
        self.x, self.y = None, None
        self.binary = True
        self.samples = None

        # Model specific settings
        if type(self.model).__name__ == 'LinearRegression':
            self.nTrials = 1

        # Input tests
        assert model is not None, 'Need to provide a model'
        if scoring is None:
            if 'Classifier' in type(model).__name__:
                self.scoring = SCORERS['accuracy']
            elif 'Regressor' in type(model).__name__:
                self.scoring = SCORERS['neg_mean_squared_error']
            else:
                raise ValueError('Model mode unknown')

    def get_params(self, trial):
        # todo support suggest log uniform
        if type(self.model).__name__ == 'LinearRegression':
            return {}
        elif type(self.model).__name__ == 'Lasso' or \
                'Ridge' in type(self.model).__name__:
            return {
                'alpha': trial.suggest_uniform('alpha', 0, 10),
            }
        elif 'SV' in type(self.model).__name__:
            return {
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto', 0.001, 0.01, 0.1, 0.5, 1]),
                'C': trial.suggest_uniform('C', 0, 10),
            }
        elif 'KNeighbors' in type(self.model).__name__:
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 5, min(50, int(self.samples / 10))),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'leaf_size': trial.suggest_int('leaf_size', 1, min(100, int(self.samples / 10))),
            }

        # Regression models
        elif type(self.model).__name__ == 'DecisionTreeRegressor':
            return {
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
                'max_depth': trial.sugest_int('max_depth', 3, min(25, int(np.log2(self.samples)))),
            }
        elif type(self.model).__name__ == 'BaggingRegressor':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 10, 250),
                'max_samples': trial.suggest_uniform('max_samples', 0.5, 1),
                'max_features': trial.suggest_uniform('max_features', 0.5, 1),
            }
        elif type(self.model).__name__ == 'CatBoostRegressor':
            return dict(n_estimators=trial.suggest_int('n_estimators', 500, 2000), verbose=0, early_stopping_rounds=100,
                        od_pval=1e-5,
                        loss_function=trial.suggest_categorical('loss_function', ['MAE', 'RMSE']),
                        learning_rate=trial.suggest_loguniform('learning_rate', 0.001, 0.5),
                        l2_leaf_reg=trial.suggest_uniform('l2_leaf_reg', 0, 10),
                        depth=trial.suggest_int('depth', 3, min(10, int(np.log2(self.samples)))),
                        min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 1, min(1000, int(self.samples / 10))),
                        grow_policy=trial.suggest_categorical('grow_policy',
                                                              ['SymmetricTree', 'Depthwise', 'Lossguide']))
        elif type(self.model).__name__ == 'GradientBoostingRegressor':
            return {
                'loss': trial.suggest_categorical('loss', ['ls', 'lad', 'huber']),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
                'max_depth': trial.suggest_int('max_depth', 3, min(10, int(np.log2(self.samples)))),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, min(1000, int(self.samples / 10))),
                'max_features': trial.suggest_uniform('max_features', 0.5, 1),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1),
            }
        elif type(self.model).__name__ == 'HistGradientBoostingRegressor':
            return {
                'loss': trial.suggest_categorical('loss', ['least_squares', 'least_absolute_deviation']),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
                'max_iter': trial.suggest_int('max_iter', 100, 250),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 30, 150),
                'max_depth': trial.suggest_int('max_depth', 3, min(10, int(np.log2(self.samples)))),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, min(1000, int(self.samples / 10))),
                'l2_regularization': trial.suggest_uniform('l2_regularization', 0, 10),
                'max_bins': trial.suggest_int('max_bins', 100, 255),
                'early_stopping': [True],
            }
        elif type(self.model).__name__ == 'RandomForestRegressor':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
                'max_depth': trial.suggest_int('max_depth', 3, min(15, int(np.log2(self.samples)))),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt']),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, min(1000, int(self.samples / 10))),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
        elif type(self.model).__name__ == 'XGBRegressor':
            param = {
                "objective": 'reg:squarederror',
                "eval_metric": "rmse",
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
                "callbacks": optuna.integration.XGBoostPruningCallback(trial, "validation-rmse"),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
            }
            if param["booster"] == "gbtree" or param["booster"] == "dart":
                param["max_depth"] = trial.suggest_int("max_depth", 1, min(10, int(np.log2(self.samples))))
                param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
                param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
                param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
                param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)
            return param
        elif type(self.model).__name__ == 'LGBMRegressor':
            return {
                'num_leaves': trial.suggest_int('num_leaves', 10, 150),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, min(1000, int(self.samples / 10))),
                'min_sum_hessian_in_leaf': trial.suggest_uniform('min_sum_hessian_in_leaf', 1e-3, 0.5),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0, 1),
                'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 1),
                'callbacks': [optuna.integration.LightGBMPruningCallback(trial, "mean_absolute_error", "valid_1")],
            }

        # Classifiers
        elif type(self.model).__name__ == 'BaggingClassifier':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 10, 250),
                'max_samples': trial.suggest_uniform('max_samples', 0.5, 1),
                'max_features': trial.suggest_uniform('max_features', 0.5, 1),
            }
        elif type(self.model).__name__ == 'CatBoostClassifier':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                "verbose": 0,
                'early_stopping_rounds': 100,
                'od_pval': 1e-5,
                'loss_function': 'Logloss' if self.binary else 'MultiClass',
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
                'l2_leaf_reg': trial.suggest_uniform('l2_leaf_reg', 0, 10),
                'depth': trial.suggest_int('depth', 1, min(10, int(np.log2(self.samples)))),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, min(1000, int(self.samples / 10))),
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            }
        elif type(self.model).__name__ == 'GradientBoostingClassifier':
            return {
                'loss': trial.suggest_categorical('loss', ['deviance', 'exponential']),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
                'max_depth': trial.suggest_int('max_depth', 3, min(15, int(np.log2(self.samples)))),
                'n_estimators': trial.suggest_int('n_estimators', 100, 250),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, min(1000, int(self.samples / 10))),
                'max_features': trial.suggest_uniform('max_features', 0.5, 1),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1),
            }
        elif type(self.model).__name__ == 'HistGradientBoostingClassifier':
            return {
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 30, 150),
                'max_depth': trial.suggest_int('max_depth', 3, min(10, int(np.log2(self.samples)))),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, min(1000, int(self.samples / 10))),
                'l2_regularization': trial.suggest_uniform('l2_regularization', 0, 10),
                'max_bins': trial.suggest_int('max_bins', 100, 255),
                'early_stopping': [True],
            }
        elif type(self.model).__name__ == 'RandomForestClassifier':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 3, min(15, int(np.log2(self.samples)))),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt']),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, min(1000, int(self.samples / 10))),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
        elif type(self.model).__name__ == 'XGBClassifier':
            param = {
                "objective": "binary:logistic" if self.binary else 'multi:softprob',
                "eval_metric": "logloss",
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
                "callbacks": optuna.integration.XGBoostPruningCallback(trial, "validation-logloss"),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
            }
            if param["booster"] == "gbtree" or param["booster"] == "dart":
                param["max_depth"] = trial.suggest_int("max_depth", 1, min(10, int(np.log2(self.samples))))
                param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
                param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
                param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
                param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)
            return param
        elif type(self.model).__name__ == 'LGBMClassifier':
            return {
                "objective": "binary" if self.binary else 'multiclass',
                "metric": trial.suggest_categorical("metric", ['binary_error', 'auc', 'average_precision',
                                                               'binary_logloss']) if self.binary else
                trial.suggest_categorical('metric', ['multi_error', 'multi_logloss', 'auc_mu']),
                "verbosity": -1,
                "boosting_type": "gbdt",
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
                "num_leaves": trial.suggest_int("num_leaves", 10, 5000),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, min(1000, int(self.samples / 10))),
                "min_gain_to_split": trial.suggest_uniform("min_gain_to_split", 0, 5),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                'callbacks': [optuna.integration.LightGBMPruningCallback(trial, "neg_log_loss", "valid_1")],
            }
        else:
            # Raise error if nothing is returned
            warnings.warn('Hyper parameter tuning not implemented for {}'.format(type(self.model).__name__))
            return {}

    def fit(self, x, y):
        if isinstance(y, pd.DataFrame):
            assert len(y.keys()) == 1, 'Multiple target columns not supported.'
            y = y[y.keys()[0]]
        assert isinstance(x, pd.DataFrame), 'X should be Pandas DataFrame'
        assert isinstance(y, pd.Series), 'Y should be Pandas Series or DataFrame'

        # Set mode
        self.binary = y.nunique() == 2
        self.samples = len(y)

        # Store
        self.x, self.y = x, y

        # Set up study
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=236868), direction='maximize')
        study.optimize(self.objective, timeout=self.timeout, n_trials=self.nTrials)

        # Parse results
        optuna_results = study.trials_dataframe()
        results = pd.DataFrame({
            'date': datetime.today().strftime('%d %b %y'),
            'model': type(self.model).__name__,
            'params': [x.params for x in study.get_trials()],
            'mean_objective': optuna_results['value'],
            'std_objective': optuna_results['user_attrs_std_value'],
            'worst_case': optuna_results['value'] - optuna_results['user_attrs_std_value'],
            'mean_time': optuna_results['user_attrs_mean_time'],
            'std_time': optuna_results['user_attrs_std_time']
        })

        return results

    def objective(self, trial):
        # Metrics
        scores = []
        times = []
        master = copy.deepcopy(self.model)

        # Cross Validation
        for t, v in self.cv.split(self.x, self.y):
            # Split data
            xt, xv, yt, yv = self.x.iloc[t], self.x.iloc[v], self.y.iloc[t], self.y.iloc[v]

            # Train model
            t_start = time.time()
            model = copy.deepcopy(master)
            model.set_params(**self.get_params(trial))
            model.fit(xt, yt)

            # Results
            scores.append(self.scoring(model, xv, yv))
            times.append(time.time() - t_start)

        # Set manual metrics
        trial.set_user_attr('mean_time', np.mean(times))
        trial.set_user_attr('std_time', np.std(times))
        trial.set_user_attr('std_value', np.std(scores))

        # Stop trail (avoid overwriting)
        if trial.number == self.nTrials:
            trial.study.stop()

        return np.mean(scores)
