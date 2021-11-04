import numpy as np
import pandas as pd
from ..Utils import getModel
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


class StackingRegressor:

    def __init__(self, **params):
        """
        Wrapper class for Stacking Regressor.

        Parameters
        ----------
        stack list[tuple]: List of tuples (model name, model object)
        params list[dict]: List of model parameters for stack
        """
        # Defaults
        self._estimator_type = 'regressor'
        self.trained = False
        self.level_one = None
        self.model = None
        self.params = params
        self.stack = []
        self.n_samples = 0
        self.n_features = 0
        self.mean = None
        self.std = None
        self.set_params(**params)

    def _add_default_models(self, stack: list) -> list:
        """
        Prepares the models stack
        """
        # Add default models
        models = [i[0] for i in stack]
        if 'KNeighborsRegressor' not in models:
            stack.append(('KNeighborsRegressor', KNeighborsRegressor()))
        if 'DecisionTreeRegressor' not in models:
            stack.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
        if 'LinearRegression' not in models:
            stack.append(('LinearRegression', LinearRegression()))
        if 'SVR' not in models and self.n_samples < 5000:
            stack.append(('SVR', SVR()))
        return stack

    def fit(self, x: pd.DataFrame, y: pd.Series):
        # Set info
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std == 0] = 1

        # Normalize
        x = (x - self.mean) / self.std

        # Create stack
        self.level_one = LinearRegression()
        self.stack = self._add_default_models(self.stack)
        self.model = ensemble.StackingRegressor(self.stack, final_estimator=self.level_one)

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
                model = getModel(model_name, mode='regression', samples=self.n_samples)
                self.stack.append((model_name, model.set_params(**param)))

            # Update otherwise
            else:
                self.stack[ind[0]][1].set_params(**param)
        return self

    def get_params(self, **args):
        """
        Returns a dictionary with all params.
        """
        return self.params

    def predict(self, x):
        assert self.trained
        return self.model.predict((x - self.mean) / self.std).reshape(-1)
