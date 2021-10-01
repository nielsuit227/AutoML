import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


class LGBMClassifier:

    def __init__(self, **params):
        """
        Light GBM wrapper
        @param params: Model parameters
        """
        default = {'verbosity': -1, 'force_col_wise': True}
        for k, v in default.items():
            if k not in params.keys():
                params[k] = v
        self.params = params
        self.default = default
        self.hasPredictProba = True
        self.set_params(**self.params)
        self.classes_ = None
        self.model = None
        self.callbacks = None
        self.trained = False
        self._estimator_type = 'classifier'

    @staticmethod
    def convert_to_dataset(x, y=None):
        # Convert input
        assert type(x) in [pd.DataFrame, pd.Series, np.ndarray], 'Unsupported data input format'
        if isinstance(x, np.ndarray) and len(x.shape) == 0:
            x = x.reshape((-1, 1))

        if y is None:
            return lgb.Dataset(x)

        else:
            assert type(y) in [pd.Series, np.ndarray], 'Unsupported data label format'
            return lgb.Dataset(x, label=y)

    def fit(self, x, y):
        assert isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame), 'X needs to be of type np.ndarray or ' \
                                                                         'pd.DataFrame'
        assert isinstance(y, np.ndarray) or isinstance(y, pd.Series), 'Y needs to be of type np.ndarray or pd.Series'
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if len(y.shape) == 2:
            y = y.reshape((-1, 1))

        # Split & Convert data
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, stratify=y, shuffle=True)
        d_train = self.convert_to_dataset(train_x, train_y)
        d_test = self.convert_to_dataset(test_x, test_y)

        # Set Attributes
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            self.params['objective'] = 'binary'
        else:
            self.params['objective'] = 'multiclass'
            self.params['num_classes'] = len(self.classes_)

        # Model training
        self.model = lgb.train(self.params,
                               d_train,
                               valid_sets=[d_train, d_test],
                               feval=self.eval_function,
                               verbose_eval=0,
                               callbacks=[self.callbacks] if self.callbacks is not None else None,
                               early_stopping_rounds=100,
                               )
        self.trained = True

    def predict(self, X):
        # todo check input data
        assert self.trained is True, 'Model not yet trained'
        prediction = self.model.predict(X)

        # Parse into most-likely class
        if len(prediction.shape) == 2:
            # MULTICLASS
            return np.argmax(prediction, axis=1)
        else:
            # BINARY
            return np.round(prediction)

    def predict_proba(self, X):
        # todo check input data
        assert self.trained is True, 'Model not yet trained'
        prediction = self.model.predict(X)

        # Parse into probabilities
        if len(prediction.shape) == 2:
            # MULTICLASS
            return prediction
        else:
            # BINARY
            return np.hstack((1 - prediction, prediction)).reshape((-1, 2), order='F')

    def set_params(self, **params):
        for k, v in self.default.items():
            if k not in params.keys():
                params[k] = v
        if 'callbacks' in params.keys():
            self.callbacks = params['callbacks']
            params.pop('callbacks')
        self.params = params
        return self

    def get_params(self, **args):
        params = copy.copy(self.params)
        if 'deep' in args:
            return params
        if self.callbacks is not None:
            params['callbacks'] = self.callbacks
        return params

    def eval_function(self, prediction, d_train):
        target = d_train.get_label()
        weight = d_train.get_weight()
        if self.params['objective'] == 'multiclass':
            prediction = prediction.reshape((-1, len(self.classes_)))

        # Return f1 score
        return "neg_log_loss", - log_loss(target, prediction, sample_weight=weight, labels=self.classes_), True

