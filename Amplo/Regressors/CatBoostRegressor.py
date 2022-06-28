import catboost
from sklearn.model_selection import train_test_split


class CatBoostRegressor:
    _estimator_type = "regressor"
    default_params = {
        "verbose": 0,
        "n_estimators": 1000,
        "allow_writing_files": False,
    }

    def __init__(self, **params):
        """
        Catboost Regressor wrapper
        """
        self.model = catboost.CatBoostRegressor()
        self.trained = False
        self.callbacks = None
        self.verbose = 0
        self.early_stopping_rounds = 100
        self.set_params(**params)

    def set_params(self, **params):
        # Add default if necessary
        for k, v in self.default_params.items():
            if k not in params:
                params[k] = v

        # Remove fit options
        if "early_stopping_rounds" in params:
            self.early_stopping_rounds = params.pop("early_stopping_rounds")
        if "verbose" in params:
            self.verbose = params.pop("verbose")

        # Update model & class
        self.model.set_params(**params)
        self.params = params
        return self

    def get_params(self, **args):
        return self.model.get_params(**args)

    def fit(self, x, y):
        # Split data
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

        # Train model
        self.model.fit(
            train_x,
            train_y,
            eval_set=[(test_x, test_y)],
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
        )

        # Set trained
        self.trained = True

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x, *args, **kwargs).reshape(-1)
