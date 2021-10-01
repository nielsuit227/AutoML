import numpy as np
from sklearn.model_selection import train_test_split
import catboost


class CatBoostClassifier:

    def __init__(self, **params):
        """
        Catboost Classifier wrapper
        """
        default = {'verbose': 0, 'n_estimators': 1000, 'allow_writing_files': False}
        for k, v in default.items():
            if k not in params.keys():
                params[k] = v
        self.default = default
        self.params = params
        self.model = catboost.CatBoostClassifier(**params)
        self.hasPredictProba = True
        self.classes_ = None
        self.trained = False
        self.callbacks = None
        self.verbose = 0
        self.early_stopping_rounds = 100
        if 'early_stopping_rounds' in params.keys():
            self.early_stopping_rounds = params.pop('early_stopping_rounds')
        if 'verbose' in params.keys():
            self.verbose = params.pop('verbose')
        self.set_params(**params)
        self._estimator_type = 'classifier'

    def set_params(self, **params):
        for k, v in self.default.items():
            if k not in params.keys():
                params[k] = v
        self.model.set_params(**params)
        return self

    def get_params(self, **args):
        return self.model.get_params(**args)

    def fit(self, X, y):
        # Split data
        train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, test_size=0.1)

        # Set Attributes
        self.classes_ = np.unique(y)

        # Train model
        self.model.fit(train_x, train_y, eval_set=[(test_x, test_y)], verbose=self.verbose,
                       early_stopping_rounds=self.early_stopping_rounds)

        # Set trained
        self.trained = True

    def predict(self, X):
        return self.model.predict(X).reshape(-1)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
