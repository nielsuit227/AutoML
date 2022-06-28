from copy import deepcopy


class PartialBoostingRegressor:
    """
    Wrapper for boosting models which limits the number of estimators being used in
    the prediction.
    """

    _estimator_type = "regressor"

    _SUPPORTED_MODELS = (
        "AdaBoostRegressor",
        "GradientBoostingRegressor",
        "LGBMRegressor",
        "XGBRegressor",
        "CatBoostRegressor",
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
        if self.model_class in ["AdaBoostRegressor", "GradientBoostingRegressor"]:
            self.model = deepcopy(model)
            self.model.estimators_ = self.model.estimators_[: self.step]
        else:
            self.model = model

    def predict(self, x):
        if self.model_class in [
            "AdaBoostRegressor",
            "GradientBoostingRegressor",
        ]:
            return self.model.predict(x)
        elif self.model_class == "LGBMRegressor":
            return self.model.predict(x, num_iteration=self.step)
        elif self.model_class == "XGBRegressor":
            return self.model.predict(x, iteration_range=(0, self.step))
        elif self.model_class == "CatBoostRegressor":
            return self.model.predict(x, ntree_end=self.step)
        else:
            raise ValueError("Incorrect model type.")

    @classmethod
    def n_estimators(cls, model):
        model_class = type(model).__name__
        if model_class in [
            "AdaBoostRegressor",
            "GradientBoostingRegressor",
        ]:
            return len(model.estimators_)
        elif model_class == "LGBMRegressor":
            return model.num_trees()
        elif model_class == "XGBRegressor":
            return model.model.num_boosted_rounds()
        elif model_class == "CatBoostRegressor":
            return model.model.tree_count_
        else:
            raise ValueError("Incorrect model type.")
