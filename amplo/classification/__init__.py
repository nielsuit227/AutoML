#  Copyright (c) 2022 by Amplo.

from amplo.classification._base import BaseClassifier
from amplo.classification.catboost import CatBoostClassifier
from amplo.classification.partial_boosting import PartialBoostingClassifier
from amplo.classification.stacking import StackingClassifier

__all__ = [
    "BaseClassifier",
    "CatBoostClassifier",
    "PartialBoostingClassifier",
    "StackingClassifier",
]
