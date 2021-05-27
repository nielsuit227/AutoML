import copy
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from Amplo.Classifiers import BaseClassifier


class XGBClassifier(BaseClassifier):

    def __init__(self, params=None):
        """
        XG Boost wrapper
        @param params: Model parameters
        """
        default = {'verbosity': 0, 'force_col_wise': True}
        super().__init__(default, params)
        self.hasPredictProba = True
        self.set_params(self.params)

    @staticmethod
    def convert_to_d_matrix(x, y=None):
        # Convert input
        assert type(x) in [pd.DataFrame, pd.Series, np.ndarray], 'Unsupported data input format'
        if isinstance(x, np.ndarray) and len(x.shape) == 0:
            x = x.reshape((-1, 1))

        if y is None:
            return xgb.DMatrix(x)

        else:
            assert type(y) in [pd.Series, np.ndarray], 'Unsupported data label format'
            return xgb.DMatrix(x, label=y)

    def fit(self, x, y):
        # Split & Convert data
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
        d_train = self.convert_to_d_matrix(train_x, train_y)
        d_test = self.convert_to_d_matrix(test_x, test_y)

        # Set attributes
        self.classes_ = np.unique(y)

        # Set objective
        if len(self.classes_) == 2:
            self.params['objective'] = 'binary:logistic'
        else:
            self.params['objective'] = 'multi:softprob'
            self.params['num_class'] = len(self.classes_)

        # Model training
        self.model = xgb.train(self.params,
                               d_train,
                               evals=[(d_test, 'validation'), (d_train, 'train')],
                               verbose_eval=0,
                               callbacks=[self.callbacks] if self.callbacks is not None else None,
                               early_stopping_rounds=100,
                               )
        self.trained = True

    def predict(self, x):
        assert self.trained is True, 'Model not yet trained'
        d_predict = self.convert_to_d_matrix(x)
        prediction = self.model.predict(d_predict)

        # Parse into most-likely class
        if len(prediction.shape) == 2:
            # MULTICLASS
            return np.argmax(prediction, axis=1)
        else:
            # BINARY
            return np.round(prediction)

    def predict_proba(self, x):
        assert self.trained is True, 'Model not yet trained'
        d_predict = self.convert_to_d_matrix(x)
        prediction = self.model.predict(d_predict)

        # Parse into probabilities
        if len(prediction.shape) == 2:
            # MULTICLASS
            return prediction
        else:
            # BINARY
            return np.hstack((1 - prediction, prediction)).reshape((-1, 2), order='F')

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
