#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Iterable, cast
from warnings import warn

import numpy as np
import pandas as pd
from requests import HTTPError

from amplo.api.storage import AzureBlobDataAPI
from amplo.utils.logging import get_root_logger

if TYPE_CHECKING:
    from logging import Logger

    from amplo.api.platform import AmploPlatformAPI

__all__ = [
    "boolean_input",
    "parse_json",
    "NpEncoder",
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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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
        "file_name": str(file_path.name),
        "full_path": str(file_path.resolve()),
        # "creation_time": os.path.getctime(str(file_path)),
        "last_modified": os.path.getmtime(str(file_path)),
    }


def _read_files_in_folders(
    folders: Iterable[str | Path],
    blob_api: AzureBlobDataAPI | None = None,
    logger: Logger | None = None,
) -> tuple[list[str], list[pd.DataFrame], list[dict[str, str | float]]]:
    """
    Use pandas to read all non-hidden and non-empty files into a DataFrame.

    Parameters
    ----------
    folders : iterable of (str or Path)
        Directory names.
    blob_api : AzureBlobDataAPI or None, optional, default: None
        If None, tries to read data from local folder, else from Azure
    logger : Logger or None, optional, default: None
        When provided, will log progress every 90 seconds.

    Returns
    -------
    data : pd.DataFrame
        All files of the folders merged into one multi-indexed DataFrame.
    metadata : list of dict of {str : str or float}
        Metadata of data.

    Warnings
    --------
    UserWarning
        When any directory is empty, or has no supported file type.
    """

    # Map folders to pathlib.Path object
    folders = list(map(Path, folders))
    folders = cast(list[Path], folders)  # type hint

    # Initialize
    file_names, data, metadata = [], [], []
    last_time_logged = time()
    for folder_count, folder in enumerate(sorted(folders)):

        # List all files
        if blob_api:
            files = list(map(Path, blob_api.ls_files(folder)))
        else:
            files = [f for f in folder.iterdir() if f.is_file()]

        # Remove hidden files
        hidden_files = [f for f in files if re.match(r"^\..*", f.name)]
        files = list(set(files) - set(hidden_files))

        # Remove unsupported file types
        unsupported_files = [f for f in files if f.suffix not in FILE_READERS]
        files = list(set(files) - set(unsupported_files))

        # Remove empty files
        if blob_api:
            empty_files = [f for f in files if blob_api.get_size(f) == 0]
        else:
            empty_files = [f for f in files if f.stat().st_size == 0]
        files = list(set(files) - set(empty_files))

        # Sanity check
        if not files:
            warn(f"Directory is empty and thus skipped: '{folder}'")
            continue

        # Read files
        for file in sorted(files):

            # read_pandas() may raise an EmptyDataError when the file has no content.
            # The try...except catches such errors and warns the user instead.
            try:
                if blob_api:
                    datum = blob_api.read_pandas(file)
                    metadatum = blob_api.get_metadata(file)
                else:
                    datum = read_pandas(file)
                    metadatum = get_file_metadata(file)
            except pd.errors.EmptyDataError:
                warn(f"Empty file detected and thus skipped: '{file}'")
                continue

            if isinstance(datum, pd.Series):
                datum = datum.to_frame()

            file_names.append(str(file))
            data.append(datum)
            metadata.append(metadatum)

        if logger and time() - last_time_logged > 90:
            last_time_logged = time()
            logger.info(f".. progress: {folder_count / len(folders) * 100:.1f} %")

    return file_names, data, metadata


def _map_datalogs_to_file_names(
    file_names: list[str],
    platform_api: AmploPlatformAPI | None = None,
    logger: Logger | None = None,
) -> list[dict]:
    """
    Get datalogs for every filename.

    Parameters
    ----------
    file_names : list of str
        Files names to get datalogs from - if available.
    platform_api : AmploPlatformAPI or None, optional, default: None
        API to get datlogs from.
    logger : Logger or None, optional, default: None
        When provided, will log progress every 90 seconds.

    Returns
    -------
    list of dict
        Datalogs for every filename.
    """

    if not platform_api:
        return []

    # It is assumed that the 6th and 5th path position of the (first) filename contains
    # the team and machine name, respectively, if you count from right to left.
    # E.g., "Team/Machine/data/Category/Issue/log_file.csv"

    # Remove path prefixes, otherwise datalogs will not be found
    file_names = ["/".join(str(fname).split("/")[-6:]) for fname in file_names]
    # Extract team and machine
    try:
        team, machine = file_names[0].split("/")[-6:-4]
    except IndexError:
        warn(f"Got an empty list of file names")
        return []

    # Get datalog for each filename
    datalogs = []
    last_time_logged = time()
    for file_count, fname in enumerate(file_names):
        try:
            datalog = platform_api.get_datalog(team, machine, fname)
        except HTTPError:
            # No matching datalog found. Do still append it to preserve the order.
            datalog = {}

        datalogs.append(datalog)

        if logger and time() - last_time_logged > 90:
            last_time_logged = time()
            logger.info(f".. progress: {file_count / len(file_names) * 100:.1f} %")

    return datalogs


def _mask_intervals(
    datalogs: list[dict],
    data: list[pd.DataFrame],
    metadata: list[dict[str, str | float]],
) -> tuple[list[pd.DataFrame], list[dict[str, str | float]]]:
    """
    Masks the data with the intervals given by the datalogs.

    Parameters
    ----------
    datalogs : list of dict
        Datalogs dictionary that should contain the keys 'selected' and 'datetime_col'.
    data : list of pd.DataFrame
        Data for splitting.
    metadata : list of dict of {str : str or float}
        Metadata of data. For each data split it will be duplicated.

    Returns
    -------
    data_out : list of pd.DataFrame
        Selected data.
    metadata_out : list of dict of {str : str or float}
        Same metadata but duplicated for every split.

    Warnings
    --------
    UserWarning
        When no valid match for the start or stop time of the data interval was found,
        i.e. when the time difference is more than 1 second.
    """

    # Initialize
    data_out = []
    metadata_out = []

    for datalog, datum, metadatum in zip(datalogs, data, metadata):
        # Get intervals and timestamp column from datalog
        intervals = datalog.get("selected", [])
        ts_col = datalog.get("datetime_col", "")

        # If no interval is to be selected, append the whole datum
        if not intervals or not ts_col:
            data_out.append(datum)
            metadata_out.append(metadatum)

        # Prevent a KeyError when ts_col column is not present
        elif ts_col not in datum.columns:
            warn(f"Cannot select intervals as the column '{ts_col}' is not present.")
            data_out.append(datum)
            metadata_out.append(metadatum)

        # Else, select and append each interval of the datum
        else:
            # Extract unix timestamps
            ts = datum[ts_col]
            if not pd.api.types.is_datetime64_any_dtype(ts):
                ts = pd.to_datetime(ts)
            ts = ts.astype(int) / 10**9  # convert to unix format

            # Extract intervals
            for interval in intervals:
                # Find closest timestamps
                ts_first, ts_last = interval
                first = (ts - ts_first).abs().argmin()
                last = (ts - ts_last).abs().argmin()

                # Ignore when time difference is too large
                if (
                    abs(ts.iloc[first] - ts_first) > 1
                    or abs(ts.iloc[last] - ts_last) > 1
                ):
                    warn(
                        f"Could not find a timestamp close enough to {ts_first=} or "
                        f"{ts_last=}. Using the whole datum for the file "
                        f"'{metadatum.get('file_name', 'unknown')}' instead."
                    )
                    data_out.append(datum)
                    metadata_out.append(metadatum)
                    continue

                # Select interval
                data_out.append(datum.iloc[first : last + 1])
                metadata_out.append(metadatum)

    return data_out, metadata_out


def _make_multiindex(
    data: list[pd.DataFrame],
    metadata: list[dict[str, str | float]],
    target_col: str = "labels",
) -> tuple[pd.DataFrame, dict[int, dict[str, str | float]]]:
    """
    Merge list of dataframes into one multiindexed one.

    Parameters
    ----------
    data : list of pd.DataFrame
        Data to be merged.
    metadata : list of dict of {str : str or float}
        Metadata of data.
    target_col : str, optional
        Target column name. Values are depicted by the folder name, by default "labels"

    Returns
    -------
    data : pd.DataFrame
        All files of the folders merged into one multi-indexed DataFrame.
    metadata : dict of {int : dict of {str : str or float}}
        Metadata of merged data.
        The first index level values of the data correspond to the keys in the metadata.

    Raises
    ------
    ValueError
        When any file already has a column named after target_col and its values are not
        equal to the folder name.
    """

    # Initialize
    index_count = 0
    data_out = []
    metadata_out = {}

    for datum, metadatum in zip(data, metadata):

        # Get file and folder name
        full_path = str(metadatum["full_path"])
        folder_name = Path(full_path).parent.name

        # Set label column
        if target_col in datum.columns and any(datum[target_col] != folder_name):
            raise ValueError(
                f"The column '{target_col}' already exists in the file '{full_path}'."
            )
        else:
            datum[target_col] = folder_name

        # Set multiindex
        index = pd.MultiIndex.from_product(
            [[index_count], datum.index.values], names=["log", "index"]
        )
        datum.set_index(index, inplace=True)

        # Add data and metadata, and increment
        data_out.append(datum)
        metadata_out[index_count] = metadatum
        index_count += 1

    # Finish: concatenate data
    return pd.concat(data_out), metadata_out


def merge_folders(
    folders: Iterable[str | Path],
    target_col: str = "labels",
    blob_api: AzureBlobDataAPI | None = None,
    platform_api: AmploPlatformAPI | None = None,
    logging: bool = False,
) -> tuple[pd.DataFrame, dict[int, dict[str, str | float]]]:
    """
    Combine log files from given directories into a multiindexed DataFrame.

    Parameters
    ----------
    folders : iterable of (str or Path)
        Directory paths to read data from.
    target_col : str, optional, default: "labels"
        Target column name. Values are depicted by the folder name
    blob_api : AzureBlobDataAPI or None, optional, default: None
        Azure api when not reading from local
    platform_api : AmploPlatformAPI or None, optional default: None
        Platform api for selecting intervals
    logging : bool, default: False
        Whether to show logging info.

    Returns
    -------
    data : pd.DataFrame
        All files of the folders merged into one multi-indexed DataFrame.
    metadata : dict of {int : dict of {str : str or float}}
        Metadata of merged data.
        The first index level values of the data correspond to the keys in the metadata.
    """

    logger = get_root_logger() if logging else None

    # Pandas read files
    logger.info("Reading files") if logger else None
    fnames, data, metadata = _read_files_in_folders(folders, blob_api, logger)

    # Select intervals when datalogs are available
    logger.info("Reading datalogs from platform") if logger else None
    datalogs = _map_datalogs_to_file_names(fnames, platform_api, logger)
    if datalogs:
        logger.info("Masking intervals from datalogs") if logger else None
        data, metadata = _mask_intervals(datalogs, data, metadata)

    # Concatenate data and make it multiindexed
    logger.info("Making multiindexed data") if logger else None
    data, metadata = _make_multiindex(data, metadata, target_col)

    return data, metadata


def merge_logs(
    parent_folder: str | Path,
    target_col: str = "labels",
    *,
    more_folders: list[str | Path] | None = None,
    azure: tuple[str, str] | bool = False,
    platform: tuple[str, str] | bool | None = None,
    logging: bool = False,
) -> tuple[pd.DataFrame, dict[int, dict[str, str | float]]]:
    """
    Combine log files of all subdirectories into a multi-indexed DataFrame.

    The function can handle logs from a local directory as well as data coming from an
    Azure blob storage. For the latter case it is furthermore capable to select
    intervals using Amplo's datalogs.

    Notes
    -----
    Make sure that each protocol is located in a subdirectory whose name represents the
    respective label.

    An exemplary directory structure of ``parent_folder``:
        ``
        parent_folder
        ├─ Label_1
        │   ├─ Log_1.*
        │   └─ Log_2.*
        ├─ Label_2
        │   └─ Log_3.*
        └─ ...
        ``

    Parameters
    ----------
    parent_folder : str or Path
        Directory that contains subdirectories with tabular data files.
    target_col : str
        Target column name. Values are depicted by the folder name.
    more_folders : list of str or Path, optional
        Additional folder names with tabular data files to append.
    azure : (str, str) or bool, default: False
        Use this parameter to indicate that data is in Azure blob storage.
        If False, it is assumed that data origins from local directory.
        If True, the AzureBlobDataAPI is initialized with default OS env variables.
        Otherwise, it will use the tuple to initialize the api.
    platform : (str, str) or bool or None, default: None
        Use this parameter for selecting data according to Amplo's datalogs.
        If None, its value is set to bool(azure).
        If False, no datalogs information will be used.
        If True, the AmploPlatformAPI is initialized with default OS env variables.
        Otherwise, it will use the tuple to initialize the api.
    logging : bool, default: False
        Whether to show logging info.

    Returns
    -------
    data : pd.DataFrame
        All files of the folders merged into one multi-indexed DataFrame.
        Multi-index names are 'log' and 'index'.
    metadata : dict of {int : dict of {str : str or float}}
        Metadata of merged data.
    """

    from amplo.api.platform import AmploPlatformAPI
    from amplo.utils import check_dtypes

    check_dtypes("parent_folder", parent_folder, (str, Path))
    check_dtypes("target_col", target_col, str)
    check_dtypes("more_folders", more_folders, (type(None), list))

    # Get azure blob client
    if not azure:
        blob_api = None
    else:
        azure = azure if not isinstance(azure, bool) else tuple()
        blob_api = AzureBlobDataAPI.from_os_env(*azure)

    # Mirror azure parameter when platform is not set
    if platform is None:
        platform = bool(azure)
    # Get amplo platform client
    if not platform:
        platform_api = None
    else:
        platform = platform if not isinstance(platform, bool) else tuple()
        platform_api = AmploPlatformAPI.from_os_env(*platform)

    # Get child folders
    if not blob_api:
        folders = [
            folder for folder in Path(parent_folder).iterdir() if folder.is_dir()
        ]
    else:
        folders = blob_api.ls_folders(parent_folder)

    # Add more_folders
    if more_folders:
        folders += more_folders

    return merge_folders(folders, target_col, blob_api, platform_api, logging)
