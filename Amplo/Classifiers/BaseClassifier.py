import numpy as np


class BaseClassifier:

    def __init__(self, default_params=None, **params):
        assert isinstance(default_params, dict), 'Provided default parameters not of type dict'
        assert isinstance(params, dict), 'Provided parameters not of type dict'
        self.defaultParams = default_params
        self.model = None
        self.hasPredictProba = False
        self.trained = False
        self.classes_ = None
        self.callbacks = None
        self.params = params if params is not None else self.defaultParams
        for key in [k for k in self.defaultParams if k not in self.params]:
            self.params[key] = self.defaultParams[key]

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def reset_weights(self):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.hasPredictProba:
            return self.model.predict_proba(X)
        else:
            raise AttributeError('{} has no predict_proba'.format(type(self.model).__name__))

    def score(self, X, y):
        return self.model.score(X, y)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model.fit(X, y)
        self.trained = True
        return self
