#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Generator
from typing import Any

__all__ = [
    "hist_search",
    "clean_feature_name",
    "check_dtypes",
    "unique_ordered_list",
]


def unique_ordered_list(seq: list[Any]):
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


def hist_search(array, value):
    """
    Binary search that finds the index in order to fulfill
    ``array[index] <= value < array[index + 1]``

    Parameters
    ----------
    array : array of float
    value : float

    Returns
    -------
    int
        Bin index of the value
    """

    # Return -1 when no bin exists
    if value < array[0] or value >= array[-1]:
        logging.debug(
            f"No bin (index) found for value {value}. "
            f"Array(Min: {array[0]}, "
            "Max: {array[-1]})"
        )
        return -1

    # Initialize min and max bin index
    low = 0
    high = len(array) - 1

    # Bin search
    countdown = 30
    while countdown > 0:
        # Count down
        countdown -= 1

        # Set middle bin index
        middle = low + (high - low) // 2

        if low == middle == high - 1:  # stop criterion
            return middle

        if value < array[middle]:  # array[low] <= value < array[middle]
            high = middle
        elif value >= array[middle]:  # array[middle] <= value < array[high]
            low = middle

    warnings.warn("Operation took too long. Returning -1 (no match).", RuntimeWarning)
    return -1


def clean_feature_name(feature_name: str | int) -> str:
    """
    Clean feature names and append "feature_" when it's a digit.

    Parameters
    ----------
    feature_name : str or int
        Feature name to be cleaned.

    Returns
    -------
    cleaned_feature_name : str
    """
    # Handle digits
    if isinstance(feature_name, int) or str(feature_name).isdigit():
        feature_name = f"feature_{feature_name}"

    # Remove non-numeric and non-alphabetic characters.
    # Assert single underscores and remove underscores in prefix and suffix.
    return re.sub("[^a-z0-9]+", "_", feature_name.lower()).strip("_")


def check_dtype(name: str, value: Any, typ: type | tuple[type, ...]) -> None:
    """
    Checks all dtype.

    Parameters
    ----------
    name : str
        Parameter name, required for properly raising the error.
    value : Any
        Parameter value to be checked.
    typ : type
        Required parameter type.

    Returns
    -------
    None

    Examples
    --------
    >>> check_dtype("var1", 123, int)

    Raises
    ------
    TypeError
        If given type constraint is not fulfilled.
    """

    if not isinstance(value, typ):
        msg = f"Invalid dtype for argument '{name}': {type(value).__name__}"
        raise TypeError(msg)


_DTCheckType = tuple[str, Any, "type | tuple[type, ...]"]


def check_dtypes(
    *checks: _DTCheckType | Generator[_DTCheckType, None, None] | list[_DTCheckType]
) -> None:
    """
    Checks all dtypes.

    Parameters
    ----------
    checks : _DTCheckType | Generator[_DTCheckType, None, None] | list[_DTCheckType]
        Tuples, generators or lists of (name, parameter, allowed types) to be checked.

    Returns
    -------
    None

    Examples
    --------
    Check a single parameter:
    >>> check_dtypes(("var1", 123, int))

    Check multiple:
    >>> check_dtypes(("var1", 123, int), ("var2", 1.0, (int, float)))  # tuples
    >>> check_dtypes(("var", var, str) for var in ["a", "b"])  # generator
    >>> check_dtypes([("var", var, str) for var in ["a", "b"]])  # list

    Raises
    ------
    TypeError
        If any given type constraint is not fulfilled.
    """

    for check in checks:
        if isinstance(check, (list, Generator)):
            check_dtypes(*check)
        else:
            check_dtype(*check)
