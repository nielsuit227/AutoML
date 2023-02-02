#  Copyright (c) 2022 by Amplo.

from amplo.regression._base import BaseRegressor
from amplo.regression.catboost import CatBoostRegressor
from amplo.regression.lgbm import LGBMRegressor
from amplo.regression.partial_boosting import PartialBoostingRegressor
from amplo.regression.stacking import StackingRegressor
from amplo.regression.xgb import XGBRegressor

__all__ = [
    "BaseRegressor",
    "CatBoostRegressor",
    "LGBMRegressor",
    "PartialBoostingRegressor",
    "StackingRegressor",
    "XGBRegressor",
]
