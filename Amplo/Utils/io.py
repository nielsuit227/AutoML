import os
import json
import warnings
from typing import Union, Tuple
from pathlib import Path

import pandas as pd


__all__ = ['boolean_input', 'parse_json', 'read_pandas', 'merge_logs']


def boolean_input(question: str) -> bool:
    x = input(question + ' [y / n]')
    if x.lower() == 'n' or x.lower() == 'no':
        return False
    elif x.lower() == 'y' or x.lower() == 'yes':
        return True
    else:
        print('Sorry, I did not understand. Please answer with "n" or "y"')
        return boolean_input(question)


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


def read_pandas(path: Union[str, Path]) -> pd.DataFrame:
    """
    Wrapper for various read functions

    Returns
    -------
    pd.DataFrame
    """
    f_ext = Path(path).suffix
    if f_ext == '.csv':
        return pd.read_csv(path)
    elif f_ext == '.json':
        return pd.read_json(path)
    elif f_ext == '.xml':
        return pd.read_xml(path)
    elif f_ext == '.feather':
        return pd.read_feather(path)
    elif f_ext == '.parquet':
        return pd.read_parquet(path)
    elif f_ext == '.stata':
        return pd.read_stata(path)
    elif f_ext == '.pickle':
        return pd.read_pickle(path)
    else:
        raise NotImplementedError('File format not supported.')


def merge_logs(path_to_folder: Union[str, Path], target: str = 'labels') -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Combine log files from given directory into a multi-indexed dataframe

    Notes
    -----
    Make sure that each protocol is located in a sub folder whose name represents the respective label.

    A directory structure example:
        |   ``path_to_folder``
        |   ``├─ Label_1``
        |   ``│   ├─ Log_1.*``
        |   ``│   └─ Log_2.*``
        |   ``├─ Label_2``
        |   ``│   └─ Log_3.*``
        |   ``└─ ...``

    Parameters
    ----------
    path_to_folder : str or Path
        Parent directory
    target : str
        Column name for target

    Returns
    -------
    data : pd.DataFrame
        All logs concatenated into one multi-indexed dataframe.
        Multi-index names are ``log`` and ``index``.
        Target column depicts the folder name.
    metadata : pd.DataFrame
        Metadata with ``folder``, ``file``, ``full_path`` and ``last_modified`` column, single-indexed
    """
    # Tests
    assert Path(path_to_folder).is_dir(), 'Invalid path to directory'
    assert Path(path_to_folder).exists(), 'Directory does not exist'
    assert isinstance(target, str), 'Target name should be a string'
    assert target != '', 'Do not use an empty string as target name'

    # Result init
    data = []
    metadata = []
    n_files = 0

    # Loop through folders
    for folder in sorted(Path(path_to_folder).iterdir()):

        # Loop through files (ignore hidden files)
        for file in sorted(folder.glob('[!.]*.*')):

            # Read df
            try:
                datum = read_pandas(file)
            except pd.errors.EmptyDataError:
                # Skip when empty
                warnings.warn(f'[AutoML] Empty file detected: {file}')
                continue

            # Set labels
            datum[target] = folder.name

            # Set index
            datum.set_index(pd.MultiIndex.from_product([[n_files], datum.index.values], names=['log', 'index']),
                            inplace=True)

            # Set metadata
            meta = pd.DataFrame({
                'folder': [folder.name], 'file': [file.name], 'full_path': [str(file.resolve())],
                'last_modified': [os.path.getmtime(str(file))]
            }, index=pd.Index([n_files], name='log'))

            # Add to list
            data.append(datum)
            metadata.append(meta)

            # Increment
            n_files += 1

    if n_files == 0:
        raise FileNotFoundError('Folder directory seems to be empty. Check whether you specified the correct path')
    elif n_files == 1:
        # Omit concatenation when only one item
        return data[0], metadata[0]
    else:
        # Concatenate dataframes
        return pd.concat(data), pd.concat(metadata)
