import catboost
import numpy as np
from sklearn.model_selection import train_test_split


class CatBoostClassifier:
    _estimator_type = "classifier"
    default_params = {
        "verbose": 0,
        "n_estimators": 1000,
        "allow_writing_files": False,
    }
    has_predict_proba = True

    def __init__(self, **params):
        """
        Catboost Classifier wrapper
        """
        self.model = catboost.CatBoostClassifier()
        self.classes_ = None
        self.trained = False
        self.callbacks = None
        self.verbose = 0
        self.early_stopping_rounds = 100
        self.use_best_model = True
        self.set_params(**params)

    def set_params(self, **params):
        # Add defaults if not present
        for k, v in self.default_params.items():
            if k not in params.keys():
                params[k] = v

        # Take out fit settings
        if "early_stopping_rounds" in params:
            self.early_stopping_rounds = params.pop("early_stopping_rounds")
        if "use_best_model" in params:
            self.use_best_model = params.pop("use_best_model")
        if "verbose" in params:
            self.verbose = params.pop("verbose")

        # Update params
        self.params = params
        self.model.set_params(**params)
        return self

    def get_params(self, *args, **kwargs):
        return self.model.get_params(*args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        # Split data
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, stratify=y, test_size=0.1
        )

        # Set Attributes
        self.classes_ = np.unique(y)

        # Train model
        self.model.fit(
            train_x,
            train_y,
            eval_set=[(test_x, test_y)],
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=self.use_best_model,
        )

        # Set trained
        self.trained = True

    def predict(self, X, *args, **kwargs):
        return self.model.predict(X, *args, **kwargs).reshape(-1)

    def predict_proba(self, X, *args, **kwargs):
        return self.model.predict_proba(X, *args, **kwargs)
