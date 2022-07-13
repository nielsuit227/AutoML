#  Copyright (c) 2022 by Amplo.

import warnings

import pandas as pd
from sklearn.feature_selection import r_regression  # pearson coefficient
from sklearn.preprocessing import LabelEncoder

from amplo.utils.util import clean_feature_name

__all__ = [
    "influx_query_to_df",
    "clean_keys",
    "check_dataframe_quality",
    "check_pearson_correlation",
]


def influx_query_to_df(result):
    df = []
    for table in result:
        parsed_records = []
        for record in table.records:
            parsed_records.append((record.get_time(), record.get_value()))
        df.append(pd.DataFrame(parsed_records, columns=["ts", record.get_field()]))
    return pd.concat(df).set_index("ts").groupby(level=0).sum()


def clean_keys(data: pd.DataFrame) -> pd.DataFrame:
    # Clean Keys
    new_keys = {}
    for key in data.keys():
        if isinstance(key, int):
            new_keys[key] = "feature_{}".format(key)
        else:
            new_keys[key] = clean_feature_name(key)
    data = data.rename(columns=new_keys)
    return data


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
