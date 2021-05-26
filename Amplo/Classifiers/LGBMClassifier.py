import copy
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


class LGBMClassifier:

    def __init__(self, params=None):
        """
        Light GBM wrapper. Uses Optuna's alternative
        """
        # Parameters
        self.model = None
        self.params = None
        self.callbacks = None
        self.trained = False

        # Parse params
        self.set_params(params)

    def fit(self, x, y):
        # Split & Convert data
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
        d_train = lgb.Dataset(train_x, label=train_y)
        d_test = lgb.Dataset(test_y, label=test_y)

        # Model training
        best_params, history = dict(), list()
        self.model = lgb.train(self.params,
                               d_train,
                               valid_sets=[d_test],
                               verbose_eval=0,
                               callbacks=[self.callbacks],
                               early_stopping_rounds=100,
                               )
        self.trained = True

    def predict(self, x):
        assert self.trained is True, 'Model not yet trained'
        # Convert if dataframe
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        # Convert if single column
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        # Convert to Dataset
        d_predict = lgb.Dataset(x)
        return self.model.predict(d_predict)

    def predict_proba(self, x):
        assert self.trained is True, 'Model not yet trained'
        # Convert if dataframe
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        # Convert if single column
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        # Convert to DMatrix
        d_predict = lgb.Dataset(x)
        self.model.predict_proba(d_predict)

    def set_params(self, params):
        if 'callbacks' in params.keys():
            self.callbacks = params['callbacks']
            params.pop('callbacks')
        self.params = params

    def get_params(self):
        params = copy.copy(self.params)
        if self.callbacks is not None:
            params['callbacks'] = self.callbacks
        return params
