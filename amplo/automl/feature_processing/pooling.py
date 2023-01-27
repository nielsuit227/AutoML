#  Copyright (c) 2022 by Amplo.

"""
Defines various pooling functions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from polars import internals as pli


def parse_column(expr: str | pli.Expr) -> pli.Expr:
    if isinstance(expr, str):
        return pli.col(expr)
    elif isinstance(expr, pli.Expr):
        return expr
    else:
        raise NotImplementedError("Expr needs to be either a str or an pli.Expr.")


def root_mean_square(column: str | pli.Expr) -> pli.Expr:
    return (parse_column(column) ** 2).mean() ** 0.5


def sum_values(column: str | pli.Expr) -> pli.Expr:
    return parse_column(column).sum()


def abs_energy(column: str | pli.Expr) -> pli.Expr:
    parsed = parse_column(column)
    return parsed.dot(parsed)


def abs_max(column: str | pli.Expr) -> pli.Expr:
    return parse_column(column).abs().max()


def n_mean_crossings(column: str | pli.Expr) -> pli.Expr:
    """
    Calculates the number of crossings of x on mean.
    A crossing is defined as two sequential values where the first value is lower than
    mean and the next is greater, or vice-versa.
    """
    diffs = (parse_column(column) - parse_column(column).mean()).sign().diff()
    return (diffs.fill_nan(0).fill_null(0) != 0).sum()


def abs_sum_of_changes(column: str | pli.Expr) -> pli.Expr:
    return parse_column(column).diff().abs().sum()


def mean_of_changes(column: str | pli.Expr) -> pli.Expr:
    return parse_column(column).diff().mean()


def abs_mean_of_changes(column: str | pli.Expr) -> pli.Expr:
    return parse_column(column).diff().abs().mean()


def cid_ce(column: str | pli.Expr, normalize: bool = True) -> pli.Expr:
    """
    Calculates an estimate for a time series complexity.
    Parameters
    ----------
    column
    normalize
    Returns
    -------
    """
    parsed = parse_column(column)
    if normalize:
        parsed = (parsed - parsed.mean()) / parsed.std()
    parsed = parsed.diff()
    return parsed.dot(parsed).sqrt()


def prominent_class(x: ArrayLike):
    return np.bincount(x, minlength=2).argmax()


# ----------------------------------------------------------------------
# Globals

POOL_FUNCTIONS = {
    # --- Basics ---
    "min": pl.min,
    "max": pl.max,
    "mean": pl.mean,
    "std": pl.std,
    "median": pl.median,
    "variance": pl.var,
    # "kurtosis": pl.Expr.kurtosis,
    # "skew": pl.Expr.skew,
    # "root_mean_square": root_mean_square,
    # "sum_values": sum_values,
    # # --- Characteristics ---
    # "entropy": pl.Expr.entropy,
    # "abs_energy": abs_energy,
    # "abs_max": abs_max,
    # # TODO: Add linear trend feature for `slope` and `stderror`. We have to
    # #  adjust our pooling function to support target data as an input.
    # #  c.f. tsfresh.feature_extraction.linear_trend
    # # "linear_trend_slope": ...,
    # # "linear_trend_stderror": ...,
    # "n_mean_crossings": n_mean_crossings,
    # # --- Difference ---
    # "abs_sum_of_changes": abs_sum_of_changes,
    # "mean_of_changes": mean_of_changes,
    # "abs_mean_of_changes": abs_mean_of_changes,
    # "cid_ce": cid_ce,
}
MORE_POOL_FUNCTIONS = {
    "prominent_class": prominent_class,
}


def get_pool_functions(requests: str | list[str] | None = None) -> dict[str, Callable]:
    """
    Get pool functions.

    Parameters
    ----------
    requests : str or List of str, optional
        If None, defaults (POOL_FUNCTIONS) are returned.

    Returns
    -------
    dict of {str: typing.Callable}
    """
    if requests is None:
        return POOL_FUNCTIONS
    elif isinstance(requests, str):
        requests = [requests]

    pools = {}
    for req in requests:
        if req in POOL_FUNCTIONS:
            pools[req] = POOL_FUNCTIONS[req]
        elif req in MORE_POOL_FUNCTIONS:
            pools[req] = MORE_POOL_FUNCTIONS[req]
        else:
            raise ValueError(f"Invalid pool function request: {req}")

    return pools
