#  Copyright (c) 2022 by Amplo.

import json
import os
from pathlib import Path
from typing import Union
from warnings import warn

import pandas as pd

__all__ = [
    "boolean_input",
    "parse_json",
    "read_pandas",
    "merge_logs",
    "get_log_metadata",
]


FILE_READERS = {
    ".csv": pd.read_csv,
    ".json": pd.read_json,
    ".xml": pd.read_xml,
    ".feather": pd.read_feather,
    ".parquet": pd.read_parquet,
    ".stata": pd.read_stata,
    ".pickle": pd.read_pickle,
}


def boolean_input(question: str) -> bool:
    x = input(question + " [y / n]")
    if x.lower() == "n" or x.lower() == "no":
        return False
    elif x.lower() == "y" or x.lower() == "yes":
        return True
    else:
        warn('Sorry, I did not understand. Please answer with "n" or "y"')
        return boolean_input(question)


def parse_json(json_string: Union[str, dict]) -> Union[str, dict]:
    if isinstance(json_string, dict):
        return json_string
    else:
        try:
            return json.loads(
                json_string.replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
                .replace("nan", "NaN")
                .replace("None", "null")
            )
        except json.decoder.JSONDecodeError:
            warn(f"Cannot validate, impassable JSON: {json_string}")
            return json_string


def read_pandas(path: Union[str, Path]) -> pd.DataFrame:
    """
    Wrapper for various read functions

    Returns
    -------
    pd.DataFrame
    """
    file_extension = Path(path).suffix
    if file_extension not in FILE_READERS:
        raise NotImplementedError(f"File format {file_extension} not supported.")
    else:
        reader = FILE_READERS[file_extension]
        return reader(path, low_memory=False)


def merge_logs(path_to_folder, target="labels"):
    r"""
    Combine log files from given directory into a multi-indexed dataframe

    Notes
    -----
    Make sure that each protocol is located in a sub folder whose name represents the
    respective label.

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
    metadata : dict
        File metadata

    Warns
    --------
    EmptyDataError
        Whenever an empty file is found.
    NotImplementedError
        Whenever a file with an unknown format is found.
    """
    # Tests
    if not Path(path_to_folder).is_dir():
        raise ValueError(f"The provided path is no directory: {path_to_folder}")
    if not Path(path_to_folder).exists():
        raise FileNotFoundError(f"Directory does not exist: {path_to_folder}")
    if not isinstance(target, str) or target == "":
        raise ValueError("Target name must be a non-empty string.")

    # Result init
    data = []

    # Get file names
    metadata = get_log_metadata(path_to_folder)

    # Loop through file paths in metadata
    for file_id in metadata:
        # Read data
        try:
            datum = read_pandas(metadata[file_id]["full_path"])
        except pd.errors.EmptyDataError:
            warn(f"Empty file: {metadata[file_id]}")
            continue
        except NotImplementedError:
            warn(f"Unknown file format: {metadata[file_id]}")

        # Set labels
        datum[target] = metadata[file_id]["folder"]

        # Set index
        datum.set_index(
            pd.MultiIndex.from_product(
                [[file_id], datum.index.values], names=["log", "index"]
            ),
            inplace=True,
        )

        # Add to list
        data.append(datum)

    if len(data) == 1:
        # Omit concatenation when only one item
        return data[0], metadata
    else:
        # Concatenate dataframes
        return pd.concat(data), metadata


def get_log_metadata(path_to_folder):
    """Get metadata of log files

    Parameters
    ----------
    path_to_folder

    Notes
    -----
    Make sure that each protocol is located in a sub folder whose name represents the
    respective label.

    A directory structure example:
        |   ``path_to_folder``
        |   ``├─ Label_1``
        |   ``│   ├─ Log_1.*``
        |   ``│   └─ Log_2.*``
        |   ``├─ Label_2``
        |   ``│   └─ Log_3.*``
        |   ``└─ ...``

    Returns
    -------
    metadata : dict
        Dictionary whose keys depict the file id (integer) and each value contains a
        dictionary with ``folder``, ``file``, ``full_path`` and ``last_modified`` key.
    """
    # Checks
    if not Path(path_to_folder).is_dir():
        raise ValueError(f"The provided path is no directory: {path_to_folder}")
    if not Path(path_to_folder).exists():
        raise FileNotFoundError(f"Directory does not exist: {path_to_folder}")

    # Init
    metadata = dict()
    file_id = 0

    # Loop through folders
    for folder in sorted(Path(path_to_folder).iterdir()):

        # Loop through files (ignore hidden files)
        for file in sorted(folder.glob("[!.]*.*")):

            # Check file
            if file.suffix not in FILE_READERS:
                warn(f"Skipped unsupported file format: {file}")
                continue
            elif file.stat().st_size == 0:
                warn(f"Skipped empty file: {file}")
                continue

            # Add to metadata
            metadata[file_id] = {
                "folder": str(folder.name),
                "file": str(file.name),
                "full_path": str(file.resolve()),
                "last_modified": os.path.getmtime(str(file)),
            }

            # Increment
            file_id += 1

    if file_id == 0:
        raise FileNotFoundError(
            "Directory seems to be empty. Check whether you specified the correct path."
        )

    return metadata
