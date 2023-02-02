#  Copyright (c) 2022 by Amplo.
from __future__ import annotations

import logging
import re
import warnings
from typing import Any

import pandas as pd
import polars as pl

__all__ = [
    "hist_search",
    "clean_feature_name",
    "clean_column_names",
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


def clean_column_names(data: pd.DataFrame | pl.DataFrame) -> dict[str, str]:
    """
    Cleans column names in place.

    Notes
    -----
    This used to take care of duplicate columns after cleaning the feature names which
    is no longer the case, and the data processor should take care of duplicate columns.

    Parameters
    ----------
    data : pd.DataFrame
        Data to be cleaned.

    Returns
    -------
    pd.DataFrame
        Same data but with cleaned column names.
    dict of {str : str}
        Dictionary which indicates the renaming.
    """
    # Make first renaming attempt. May create duplicated names.
    renaming = pd.Series({old: clean_feature_name(old) for old in data.columns})
    return data.rename(columns=renaming), renaming.to_dict()


def check_dtypes(*arg: tuple[str, Any, type | tuple[type, ...]]):
    """
    Checks all dtypes of given list.

    Parameters
    ----------
    name : str
    value : Any
    typ : type | tuple[type, ...]

    Returns
    -------
    None

    Examples
    --------
    Check a parameter:
    >>> check_dtypes("var1", 123, int)  # tuple

    Raises
    ------
    ValueError
        If any given constraint is not fulfilled.
    """

    def check_dtype(name: str, value: Any, typ: type | tuple[type, ...]):
        if not isinstance(value, typ):
            msg = f"Invalid dtype for argument `{name}`: {type(value).__name__}"
            if isinstance(typ, tuple):
                msg += f", expected {', '.join(t.__name__ for t in typ)}"
            else:
                msg += f", expected {typ.__name__}"
            raise TypeError(msg)

    if isinstance(arg[0], str):
        check_dtype(arg[0], arg[1], arg[2])
    else:
        for check in arg:
            check_dtype(check[0], check[1], check[2])
