import re
import pandas as pd


__all__ = ['influx_query_to_df', 'clean_keys', 'check_dataframe_quality']


def influx_query_to_df(result):
    df = []
    for table in result:
        parsed_records = []
        for record in table.records:
            parsed_records.append((record.get_time(), record.get_value()))
        df.append(pd.DataFrame(parsed_records, columns=['ts', record.get_field()]))
    return pd.concat(df).set_index('ts').groupby(level=0).sum()


def clean_keys(data: pd.DataFrame) -> pd.DataFrame:
    # Clean Keys
    new_keys = {}
    for key in data.keys():
        if isinstance(key, int):
            new_keys[key] = 'feature_{}'.format(key)
        else:
            new_keys[key] = re.sub('[^a-z0-9\n]', '_', str(key).lower()).replace('__', '_')
    data = data.rename(columns=new_keys)
    return data


def check_dataframe_quality(data: pd.DataFrame) -> bool:
    assert not data.isna().any().any()
    assert not data.isnull().any().any()
    assert not (data.dtypes == object).any().any()
    assert not (data.dtypes == str).any().any()
    assert data.max().max() < 1e38 and data.min().min() > -1e38
    return True
