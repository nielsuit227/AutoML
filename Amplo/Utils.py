import re
import json
import pandas as pd
from typing import Union


def influx_query_to_df(result):
    df = []
    for table in result:
        parsed_records = []
        for record in table.records:
            parsed_records.append((record.get_time(), record.get_value()))
        df.append(pd.DataFrame(parsed_records, columns=['ts', record.get_field()]))
    return pd.concat(df).set_index('ts').groupby(level=0).sum()


def getModel(model_str, **args):
    from Amplo.AutoML.Modeller import Modeller
    try:
        return [mod for mod in Modeller(**args).return_models() if type(mod).__name__ == model_str][0]
    except IndexError:
        raise IndexError('Model not found.')


def histSearch(array, value):
    low = 0
    high = len(array) - 1
    while low < high:
        middle = low + (high - low) // 2
        if value > array[middle + 1]:
            low = middle + 1
        elif value < array[middle]:
            high = middle
        else:
            return middle
    return -1


def boolean_input(question: str) -> bool:
    x = input(question + ' [y / n]')
    if x.lower() == 'n' or x.lower() == 'no':
        return False
    elif x.lower() == 'y' or x.lower() == 'yes':
        return True
    else:
        print('Sorry, I did not understand. Please answer with "n" or "y"')
        return boolean_input(question)


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


def parse_json(json_string: Union[str, dict]) -> Union[str, dict]:
    if isinstance(json_string, dict):
        return json_string
    else:
        try:
            return json.loads(json_string
                              .replace("'", '"')
                              .replace("True", "true")
                              .replace("False", "false")
                              .replace("nan", "NaN")
                              .replace("None", "null"))
        except json.decoder.JSONDecodeError:
            print('[AutoML] Cannot validate, impassable JSON.')
            print(json_string)
            return json_string


def check_dataframe_quality(data: pd.DataFrame) -> bool:
    assert not data.isna().any().any()
    assert not data.isnull().any().any()
    assert not (data.dtypes == object).any().any()
    assert not (data.dtypes == str).any().any()
    assert data.max().max() < 1e38 and data.min().min() > -1e38
    return True
