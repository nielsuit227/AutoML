import copy
from datetime import datetime
import joblib
import os
import time
from typing import TypeVar

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from Amplo.Classifiers import CatBoostClassifier
from Amplo.Classifiers import LGBMClassifier
from Amplo.Classifiers import XGBClassifier
from Amplo.Regressors import CatBoostRegressor
from Amplo.Regressors import LGBMRegressor
from Amplo.Regressors import XGBRegressor


__all__ = ['ClassificationType', 'Modeller', 'ModelType', 'RegressionType']


ClassificationType = TypeVar(
    'ClassificationType', CatBoostClassifier, ensemble.BaggingClassifier,
    linear_model.RidgeClassifier, LGBMClassifier, svm.SVC, XGBClassifier)
RegressionType = TypeVar(
    'RegressionType', CatBoostRegressor, ensemble.BaggingRegressor,
    linear_model.LinearRegression, LGBMRegressor, svm.SVR, XGBRegressor)
ModelType = TypeVar(
    'ModelType', CatBoostClassifier, CatBoostRegressor,
    ensemble.BaggingClassifier, ensemble.BaggingRegressor,
    linear_model.LinearRegression, linear_model.RidgeClassifier,
    LGBMClassifier, LGBMRegressor, svm.SVC, svm.SVR, XGBClassifier,
    XGBRegressor)


class Modeller:
    """
    Modeller for classification or regression tasks.

    Supported models:
        - linear models from ``scikit-learn``
        - random forest from ``scikit-learn``
        - bagging model from ``scikit-learn``
        - ~~gradient boosting from ``scikit-learn``~~
        - ~~histogram-based gradient boosting from ``scikit-learn``~~
        - XGBoost from DMLC
        - Catboost from Yandex
        - LightGBM from Microsoft

    Parameters
    ----------
    mode : str
        Model mode. Either `regression` or `classification`.
    shuffle : bool
        Whether to shuffle samples from training / validation.
    n_splits : int
        Number of cross-validation splits.
    objective : str
        Performance metric to optimize. Must be a valid string for
        `sklearn.metrics.SCORERS`.
    samples : int
        Hypothetical number of samples in dataset. Useful to manipulate behavior
        of `return_models()` function.
    needs_proba : bool
        Whether the modelling needs a probability.
    folder : str
        Folder to store models and/or results.
    dataset : str
        Name of feature set. For documentation purposes only.
    store_models : bool
        Whether to store the trained models. If true, `folder` must be
        specified to take effect.
    store_results : bool
        Whether to store the results. If true, `folder` must be specified.

    See Also
    --------
    [Sklearn scorers](https://scikit-learn.org/stable/modules/model_evaluation.html
    """

    def __init__(self,
                 mode='regression',
                 shuffle=False,
                 n_splits=3,
                 objective='accuracy',
                 samples=None,
                 needs_proba=True,
                 folder='',
                 dataset='set_0',
                 store_models=False,
                 store_results=True):
        # Test
        assert mode in ['classification', 'regression'], 'Unsupported mode'
        assert isinstance(shuffle, bool)
        assert isinstance(n_splits, int)
        assert 2 < n_splits < 10, 'Reconsider your number of splits'
        assert isinstance(objective, str)
        assert objective in metrics.SCORERS.keys(), \
            'Pick scorer from sklearn.metrics.SCORERS: \n{}'.format(list(metrics.SCORERS.keys()))
        assert isinstance(samples, int) or samples is None
        assert isinstance(folder, str)
        assert isinstance(dataset, str)
        assert isinstance(store_models, bool)
        assert isinstance(store_results, bool)

        # Parameters
        self.objective = objective
        self.scoring = metrics.SCORERS[objective]
        self.mode = mode
        self.shuffle = shuffle
        self.cvSplits = n_splits
        self.samples = samples
        self.dataset = str(dataset)
        self.storeResults = store_results
        self.storeModels = store_models
        self.results = pd.DataFrame(columns=['date', 'model', 'dataset', 'params', 'mean_objective', 'std_objective',
                                             'mean_time', 'std_time'])

        # Folder
        self.folder = folder if len(folder) == 0 or folder[-1] == '/' else folder + '/'
        if (store_results or store_models) and self.folder != '':
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

        self.needsProba = needs_proba

    def fit(self, x, y):
        # Copy number of samples
        self.samples = len(y)

        # Regression
        if self.mode == 'regression':
            cv = KFold(n_splits=self.cvSplits, shuffle=self.shuffle)
            return self._fit(x, y, cv)

        # Classification
        if self.mode == 'classification':
            cv = StratifiedKFold(n_splits=self.cvSplits, shuffle=self.shuffle)
            return self._fit(x, y, cv)

    def return_models(self):
        """
        Get all models that are considered appropriate for training.

        Returns
        -------
        list of ModelType
            Models that apply for given dataset size and mode.
        """
        models = []

        # All classifiers
        if self.mode == 'classification':
            # The thorough ones
            if self.samples < 25000:
                models.append(svm.SVC(kernel='rbf', probability=self.needsProba))
                models.append(ensemble.BaggingClassifier())
                # models.append(ensemble.GradientBoostingClassifier()) == XG Boost
                models.append(XGBClassifier())

            # The efficient ones
            else:
                # models.append(ensemble.HistGradientBoostingClassifier()) == LGBM
                models.append(LGBMClassifier())

            # And the multifaceted ones
            if not self.needsProba:
                models.append(linear_model.RidgeClassifier())
            models.append(CatBoostClassifier())
            models.append(ensemble.RandomForestClassifier())

        elif self.mode == 'regression':
            # The thorough ones
            if self.samples < 25000:
                models.append(svm.SVR(kernel='rbf'))
                models.append(ensemble.BaggingRegressor())
                # models.append(ensemble.GradientBoostingRegressor()) == XG Boost
                models.append(XGBRegressor())

            # The efficient ones
            else:
                # models.append(ensemble.HistGradientBoostingRegressor()) == LGBM
                models.append(LGBMRegressor())

            # And the multifaceted ones
            models.append(linear_model.LinearRegression())
            models.append(CatBoostRegressor())
            models.append(ensemble.RandomForestRegressor())

        return models

    def _fit(self, x, y, cross_val):
        # Convert to NumPy
        x = np.array(x)
        y = np.array(y).ravel()

        # Data
        print('[AutoML] Splitting data (shuffle=%s, splits=%i, features=%i)' %
              (str(self.shuffle), self.cvSplits, len(x[0])))

        if self.storeResults and 'Initial_Models.csv' in os.listdir(self.folder):
            self.results = pd.read_csv(self.folder + 'Initial_Models.csv')
            for i in range(len(self.results)):
                self.print_results(self.results.iloc[i])

        else:

            # Models
            self.models = self.return_models()

            # Loop through models
            for master_model in self.models:

                # Time & loops through Cross-Validation
                val_score = []
                train_score = []
                train_time = []
                for t, v in cross_val.split(x, y):
                    t_start = time.time()
                    xt, xv, yt, yv = x[t], x[v], y[t], y[v]
                    model = copy.copy(master_model)
                    model.fit(xt, yt)
                    val_score.append(self.scoring(model, xv, yv))
                    train_score.append(self.scoring(model, xt, yt))
                    train_time.append(time.time() - t_start)

                # Append results
                result = {
                    'date': datetime.today().strftime('%d %b %y'),
                    'model': type(model).__name__,
                    'dataset': self.dataset,
                    'params': model.get_params(),
                    'mean_objective': np.mean(val_score),
                    'std_objective': np.std(val_score),
                    'worst_case': np.mean(val_score) - np.std(val_score),
                    'mean_time': np.mean(train_time),
                    'std_time': np.std(train_time)
                }
                self.results = pd.concat([self.results, pd.Series(result).to_frame().T], ignore_index=True)
                self.print_results(result)

                # Store model
                if self.storeModels:
                    joblib.dump(model, self.folder + type(model).__name__ + '_{:.4f}.joblib'.format(np.mean(val_score)))

            # Store CSV
            if self.storeResults:
                self.results.to_csv(self.folder + 'Initial_Models.csv')

        # Return results
        return self.results

    def print_results(self, result):
        print('[AutoML] {} {}: {} \u00B1 {}, training time: {:.1f} s'.format(
            result['model'].ljust(30), self.objective, f"{result['mean_objective']:.4f}".ljust(15),
            f"{result['std_objective']:.4f}".ljust(15), result['mean_time']))
