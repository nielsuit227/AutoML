#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable
from warnings import warn

import pandas as pd

__all__ = [
    "boolean_input",
    "parse_json",
    "read_pandas",
    "get_file_metadata",
    "merge_folders",
    "merge_logs",
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


def parse_json(json_string: str | dict) -> str | dict:
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


def read_pandas(path: str | Path) -> pd.DataFrame:
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


def get_file_metadata(file_path: str | Path) -> dict[str, str | float]:
    """
    Get file metadata from given path.

    Parameters
    ----------
    file_path : str or Path
        File path.

    Returns
    -------
    dict of {str: str or float}
        File metadata.

    Raises
    ------
    FileNotFoundError
        When the path does not exist.
    IsADirectoryError
        When the path resolves a directory, not a file.
    """

    from amplo.utils import check_dtypes

    check_dtypes("file_path", file_path, (str, Path))

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: '{file_path}'")
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is not a file: '{file_path}'")

    return {
        "folder": str(file_path.parent.name),
        "file": str(file_path.name),
        "full_path": str(file_path.resolve()),
        "last_modified": os.path.getmtime(str(file_path)),
    }


def merge_folders(
    folders: Iterable[str | Path], target_col: str = "labels"
) -> tuple[pd.DataFrame, dict[int, dict[str, str | float]]]:
    """
    Combine log files from given directories into a multi-indexed DataFrame.

    Parameters
    ----------
    folders : iterable of str or Path
        Folder names.
    target_col : str
        Target column name. Values are depicted by the folder name.

    Returns
    -------
    data : pd.DataFrame
        All files of the folders merged into one multi-indexed DataFrame.
        Multi-index names are 'log' and 'index'.
    metadata : dict of {int : dict of {str : str or float}}
        Metadata of merged data.

    Raises
    ------
    FileNotFoundError
        When any given folder path does not exist or is empty.
    NotADirectoryError
        When any given folder path is not a directory.
    ValueError
        When any file already has a column named after ``target_col``.
    """

    from amplo.utils import check_dtypes

    if not hasattr(folders, "__iter__"):
        raise ValueError(f"Parameter `folders` is not an iterable.")
    check_dtypes("target_col", target_col, str)

    # Initialize
    data, metadata = [], {}
    counter = 0

    # Loop through folders
    folders = sorted(Path(folder) for folder in folders)
    for folder in folders:

        # Sanity checks
        if not folder.exists():
            raise FileNotFoundError(f"Directory does not exist: '{folder}'")
        if not folder.is_dir():
            raise NotADirectoryError(f"Path is not a directory: '{folder}'")
        files = sorted(folder.glob("[!.]*.*"))  # ignore hidden files
        if not files:
            raise FileNotFoundError(f"Directory seem so be empty: {folder}")

        # Read files in folder
        for file in files:

            # Skip bad files
            if file.suffix not in FILE_READERS:
                warn(f"Skipped unsupported file format: '{file}'")
                continue
            if file.stat().st_size == 0:
                warn(f"Skipped empty file: '{file}'")
                continue

            # Read data
            datum = read_pandas(file)
            metadatum = get_file_metadata(file)

            # Set labels and index
            if target_col in datum.columns:
                raise ValueError(
                    f"The target name '{target_col}' already exists in the data columns"
                    f" of file '{file}'."
                )
            datum[target_col] = folder.name

            # Set index
            index = pd.MultiIndex.from_product(
                [[counter], datum.index.values], names=["log", "index"]
            )
            datum.set_index(index, inplace=True)

            # Add data and metadata, and increment
            data.append(datum)
            metadata[counter] = metadatum
            counter += 1

    # Finish: concatenate data
    if len(data) == 0:
        raise FileNotFoundError(f"Directories seem to be empty: {folders}")
    data = pd.concat(data)

    return data, metadata


def merge_logs(
    parent_folder: str | Path,
    target_col: str = "labels",
    *,
    more_folders: list[str | Path] = None,
) -> tuple[pd.DataFrame, dict[int, dict[str, str | float]]]:
    """
    Combine log files of all subdirectories into a multi-indexed DataFrame.

    Notes
    -----
    Make sure that each protocol is located in a subdirectory whose name represents the
    respective label.

    An exemplary directory structure of ``parent_folder``:
        |   ``parent_folder``
        |   ``├─ Label_1``
        |   ``│   ├─ Log_1.*``
        |   ``│   └─ Log_2.*``
        |   ``├─ Label_2``
        |   ``│   └─ Log_3.*``
        |   ``└─ ...``

    Parameters
    ----------
    parent_folder : str or Path
        Directory that contains subdirectories with tabular data files.
    target_col : str
        Target column name. Values are depicted by the folder name.
    more_folders : list of str or Path, optional
        Additional folder names with tabular data files to append.

    Returns
    -------
    data : pd.DataFrame
        All files of the folders merged into one multi-indexed DataFrame.
        Multi-index names are 'log' and 'index'.
    metadata : dict of {int : dict of {str : str or float}}
        Metadata of merged data.

    Raises
    ------
    FileNotFoundError
        When a folder does not exist.
    NotADirectoryError
        When a folder is not a directory.
    """

    from amplo.utils import check_dtypes

    check_dtypes("parent_folder", parent_folder, (str, Path))
    check_dtypes("target_col", target_col, str)
    check_dtypes("more_folders", more_folders, (type(None), list))

    parent_folder = Path(parent_folder)

    if not parent_folder.exists():
        raise FileNotFoundError(f"Directory does not exist: '{parent_folder}'")
    if not parent_folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: '{parent_folder}'")

    folders = [folder for folder in parent_folder.iterdir() if folder.is_dir()]
    if more_folders:
        folders.extend(more_folders)

    return merge_folders(folders, target_col)
