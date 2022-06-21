from typing import TypeVar
from Amplo.GridSearch.BaseGridSearch import BaseGridSearch
from Amplo.GridSearch.HalvingGridSearch import HalvingGridSearch
from Amplo.GridSearch.OptunaGridSearch import OptunaGridSearch


GridSearchType = TypeVar('GridSearchType', BaseGridSearch, HalvingGridSearch,
                         OptunaGridSearch)
