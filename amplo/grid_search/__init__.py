#  Copyright (c) 2022 by Amplo.

from typing import TypeVar

from amplo.grid_search.exhaustive import ExhaustiveGridSearch
from amplo.grid_search.halving import HalvingGridSearch
from amplo.grid_search.optuna import OptunaGridSearch

GridSearchType = TypeVar(
    "GridSearchType", ExhaustiveGridSearch, HalvingGridSearch, OptunaGridSearch
)
