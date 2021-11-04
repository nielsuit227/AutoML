import numpy as np
from ..Utils import getModel
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


class StackingClassifier:

    def __init__(self, **params):
        """
        Wrapper class for Stacking Classifier.

        Parameters
        ----------
        stack list[tuple]: List of tuples (model name, model object)
        params list[dict]: List of model parameters for stack
        """
        # Defaults
        self._estimator_type = 'classifier'
        self.trained = False
        self.hasPredictProba = True
        self.level_one = None
        self.model = None
        self.classes_ = None
        self.params = params
        self.stack = []
        self.mean = None
        self.std = None
        self.n_samples = 0
        self.n_features = 0
        self.set_params(**params)

    def _add_default_models(self, stack: list) -> list:
        """
        Prepares the models stack
        """
        # Add default models
        models = [i[0] for i in stack]
        if 'KNeighborsClassifier' not in models:
            stack.append(('KNeighborsClassifier', KNeighborsClassifier()))
        if 'DecisionTreeClassifier' not in models:
            stack.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
        if 'LogisticRegression' not in models:
            stack.append(('LogisticRegression', LogisticRegression()))
        if 'GaussianNB' not in models:
            stack.append(('GaussianNB', GaussianNB()))
        if 'SVC' not in models and self.n_samples < 5000:
            stack.append(('SVC', SVC()))
        return stack

    def fit(self, x, y):
        # Set info
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        self.classes_ = np.unique(y)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std == 0] = 1

        # Normalize
        x = (x - self.mean) / self.std

        # Set level one
        solver = 'lbfgs'
        if self.n_samples > 10000 or self.n_features > 100:
            solver = 'sag'
        self.level_one = LogisticRegression(max_iter=2000, solver=solver)

        # Create stack
        self.stack = self._add_default_models(self.stack)
        self.model = ensemble.StackingClassifier(self.stack, final_estimator=self.level_one)

        # Fit
        self.model.fit(x, y)

        # Set flag
        self.trained = True

    def set_params(self, **params):
        """
        Set params for the models in the stack

        Parameters
        ----------
        params dict: Nested dictionary, first keys are model names, second params
        """
        # Overwrite old params
        for k, v in params.items():
            self.params[k] = v

        # Set default
        if 'n_samples' in params:
            self.n_samples = params['n_samples']
            params.pop('n_samples')
        if 'n_features' in params:
            self.n_features = params['n_features']
            params.pop('n_features')

        for model_name, param in params.items():
            # Get index
            ind = [i for i, x in enumerate(self.stack) if x[0] == model_name]

            # Add if not in stack
            if len(ind) == 0:
                model = getModel(model_name, mode='classification', samples=self.n_samples)
                self.stack.append((model_name, model.set_params(**param)))

            # Update otherwise
            else:
                self.stack[ind[0]][1].set_params(**param)
        return self

    def get_params(self, **args):
        """
        Returns a dictionary with all params.
        """
        self.params['n_samples'] = self.n_samples
        self.params['n_features'] = self.n_features
        return self.params

    def predict(self, x):
        assert self.trained
        return self.model.predict((x - self.mean) / self.std).reshape(-1)

    def predict_proba(self, x):
        assert self.trained
        return self.model.predict_proba((x - self.mean) / self.std)
