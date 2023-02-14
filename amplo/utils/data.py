#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import warnings

import pandas as pd
import polars as pl
from sklearn.feature_selection import r_regression  # pearson coefficient
from sklearn.preprocessing import LabelEncoder

from amplo.utils.util import check_dtypes

__all__ = [
    "influx_query_to_df",
    "check_dataframe_quality",
    "check_pearson_correlation",
    "pandas_to_polars",
    "polars_to_pandas",
]


def influx_query_to_df(result):
    df = []
    for table in result:
        parsed_records = []
        for record in table.records:
            parsed_records.append((record.get_time(), record.get_value()))
        df.append(
            pd.DataFrame(parsed_records, columns=["ts", table.records[0].get_field()])
        )
    return pd.concat(df).set_index("ts").groupby(level=0).sum()


def check_dataframe_quality(data: pd.DataFrame) -> bool:
    if data.isna().any().any():
        warnings.warn("Data contains NaN.")
    elif data.isnull().any().any():
        warnings.warn("Data contains null.")
    elif (data.dtypes == object).any().any():
        warnings.warn("Data contains dtype 'object', which is ambiguous.")
    elif (data.dtypes == str).any().any():
        warnings.warn("Data contains dtype 'str', which is ambiguous.")
    elif data.max().max() > 1e38 or data.min().min() < -1e38:
        warnings.warn("Data contains values larger than float32 (1e38).")
    else:
        return True
    return False


def check_pearson_correlation(features: pd.DataFrame, labels: pd.Series) -> bool:
    if labels.dtype == "object":
        labels = LabelEncoder().fit_transform(labels)
    pearson_corr = r_regression(features, labels)
    if abs(pearson_corr).mean() > 0.5:
        return False
    else:
        return True


def pandas_to_polars(
    data: pd.DataFrame, include_index: bool = True, index_prefix: None | str = None
) -> tuple[pl.DataFrame, dict[str, str]]:
    """
    Convert pandas to polars DataFrame.

    Notes
    -----
    Polars removes the indices when calling `pl.from_pandas(data)`. This function does
    per default include the index (include_index=True) and indicates the index with the
    second return argument (index_names).

    Polars also does not allow duplicate column names. Moving indices to the columns may
    introduce such duplicates. Additionally defining an index_prefix may help to avoid
    such collisions.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas object to be converted
    include_index : bool, optional
        Whether to include the index for conversion, by default True
    index_prefix : None | str, optional
        When `include_index` is True, will rename the index columns with the given
        prefix (e.g., index_prefix_1, index_prefix_2, ...), by default None

    Returns
    -------
    pl_data : pl.DataFrame
        Converted polars object.
    index_names : dict[str, str]
        Rename dictionary for the index names.
    """

    check_dtypes(
        ("data", data, pd.DataFrame),
        ("include_index", include_index, bool),
        ("index_prefix", index_prefix, str),
    )

    # Convert to polars without index
    if not include_index:
        return pl.from_pandas(data), {}

    # Get original and set new (work) index names
    orig_index_names = list(data.index.names)
    work_index_names = [f"{index_prefix}_{i}" for i in range(len(orig_index_names))]
    index_names = dict(zip(work_index_names, orig_index_names))
    # Conversion
    pl_data = pl.from_pandas(data.reset_index(names=work_index_names))

    return pl_data, index_names


def polars_to_pandas(
    data: pl.DataFrame, index_names: None | list[str] | dict[str, str] = None
) -> pd.DataFrame:
    """
    Convert polars to pandas DataFrame.

    Parameters
    ----------
    data : pl.DataFrame
        Polars object to be converted.
    index_names : None | list[str] | dict[str, str], optional
        As polars does not support indices, we recover it from its column(s).
        If None, no index is recovered.
        If list, given columns are recovered (w/o renaming them).
        If dict, values() are assumed to contain column names to recover index from and
        keys() will be the new names for it, by default None

    Returns
    -------
    pd_data : pd.DataFrame
        Converted pandas object.
    """

    check_dtypes(
        ("data", data, pl.DataFrame),
        ("index_names", index_names, (type(None), dict, list)),
    )

    # Convert to pandas
    pd_data = data.to_pandas()

    # Restore index names
    #  Skip when `index_names` is empty. Otw `pd.set_index` will raise an error.
    if index_names and isinstance(index_names, list):
        orig_index_names = work_index_names = index_names
    elif index_names and isinstance(index_names, dict):
        orig_index_names = list(index_names.values())
        work_index_names = list(index_names.keys())
    else:
        # Skip index restoral
        return pd_data

    # Restore index (optional)
    pd_data.set_index(work_index_names, inplace=True)
    pd_data.index.names = orig_index_names

    return pd_data
