from copy import deepcopy


class PartialBoostingClassifier:
    """
    Wrapper for boosting models which limits the number of estimators being used in
    the prediction.
    """

    _estimator_type = "classifier"

    _SUPPORTED_MODELS = (
        "AdaBoostClassifier",
        "GradientBoostingClassifier",
        "LGBMClassifier",
        "XGBClassifier",
        "CatBoostClassifier",
    )

    def __init__(self, model, step):
        """
        Construct wrapper for model with `predict` and `predict_proba` methods.

        Parameters
        ----------
        model
            boosting model to wrap.
        step : int
            Number of iterations/estimators to limit the model on predictions.
        """
        self.model_class = type(model).__name__
        self.step = step
        if self.model_class in ["AdaBoostClassifier", "GradientBoostingClassifier"]:
            self.model = deepcopy(model)
            self.model.estimators_ = self.model.estimators_[: self.step]
        else:
            self.model = model

    @property
    def classes_(self):
        return self.model.classes_

    def predict_proba(self, x):
        if self.model_class in ["AdaBoostClassifier", "GradientBoostingClassifier"]:
            return self.model.predict_proba(x)
        elif self.model_class == "LGBMClassifier":
            return self.model.predict_proba(x, num_iteration=self.step)
        elif self.model_class == "XGBClassifier":
            return self.model.predict_proba(x, iteration_range=(0, self.step))
        elif self.model_class == "CatBoostClassifier":
            return self.model.predict_proba(x, ntree_end=self.step)
        else:
            raise ValueError("Incorrect model type.")

    def predict(self, x):
        if self.model_class in [
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
        ]:
            return self.model.predict(x)
        elif self.model_class == "LGBMClassifier":
            return self.model.predict(x, num_iteration=self.step)
        elif self.model_class == "XGBClassifier":
            return self.model.predict(x, iteration_range=(0, self.step))
        elif self.model_class == "CatBoostClassifier":
            return self.model.predict(x, ntree_end=self.step)
        else:
            raise ValueError("Incorrect model type.")

    @classmethod
    def n_estimators(cls, model):
        model_class = type(model).__name__
        if model_class in [
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
        ]:
            return len(model.estimators_)
        elif model_class == "LGBMClassifier":
            return model.model.num_trees()
        elif model_class == "XGBClassifier":
            return model.model.num_boosted_rounds()
        elif model_class == "CatBoostClassifier":
            return model.model.tree_count_
        else:
            raise ValueError("Incorrect model type.")
