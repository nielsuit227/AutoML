#  Copyright (c) 2022 by Amplo.

import sys
from collections import deque
from collections.abc import Mapping, Set
from numbers import Number
from typing import Any

import pandas as pd

__all__ = ["getsize"]


ZERO_DEPTH_BASES = (str, bytes, Number, range, bytearray)


def getsize(obj_0: Any) -> int:
    # Credits: https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python  # noqa: E501
    """
    Recursively iterate to sum size of object & members.

    Parameters
    ----------
    obj_0 : Any
        Object whose size is estimated recursively
    """
    _seen_ids = set()

    def inner(obj: Any) -> int:
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)

        try:
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.memory_usage(deep=True)

            size = sys.getsizeof(obj)
            if isinstance(obj, ZERO_DEPTH_BASES):
                pass  # bypass remaining control flow and return
            elif isinstance(obj, (tuple, list, Set, deque)):
                size += sum(inner(i) for i in obj)
            elif isinstance(obj, Mapping) or hasattr(obj, "items"):
                size += sum(inner(k) + inner(v) for k, v in getattr(obj, "items")())
            # Check for custom object instances - may subclass above too
            if hasattr(obj, "__dict__"):
                size += inner(vars(obj))
            if hasattr(obj, "__slots__"):  # can have __slots__ with __dict__
                size += sum(
                    inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s)
                )
            return size
            # TODO: figure out what can error here.
        except:  # noqa: E722
            return 0

    return inner(obj_0)
