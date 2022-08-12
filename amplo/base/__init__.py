#  Copyright (c) 2022 by Amplo.

"""
Base class repository of Amplo.
"""

from amplo.base.exceptions import NotFittedError
from amplo.base.objects import (
    BaseEstimator,
    BaseObject,
    BasePredictor,
    BaseTransformer,
    LoggingMixin,
)
