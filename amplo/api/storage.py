#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import io
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
import pytz
import requests.exceptions
from azure.storage.blob import BlobServiceClient

from amplo.utils.util import check_dtypes, deprecated

if TYPE_CHECKING:
    from azure.storage.blob import BlobClient, ContainerClient

__all__ = ["AzureSynchronizer"]


_AZURE_CLIENT_NAME = "amploplatform"
_AZURE_CONNECTION_STR_OS = "AZURE_STORAGE_STRING"


class AzureBlobDataAPI:
    """
    Helper class for handling data from an Azure blob storage.

    Parameters
    ----------
    client : str
        Container client name.
    connection_str : str
        Connection string for given container client.
    """

    def __init__(self, client: str, connection_str: str):
        check_dtypes(("client", client, str), ("connection_str", connection_str, str))
        self.client = client
        self.connection_str = connection_str
        bsc = BlobServiceClient.from_connection_string(connection_str)
        self._container: ContainerClient = bsc.get_container_client(client)

    def __repr__(self):
        """
        Readable string representation of the class.
        """

        return f"{self.__class__.__name__}({self.client})"

    @classmethod
    def from_os_env(
        cls, client: str | None = None, connection_str_os: str | None = None
    ) -> AzureBlobDataAPI:
        """
        Instantiate the class using os environment strings.

        Parameters
        ----------
        client : str, optional, default: _AZURE_CLIENT_NAME
            Container client name.
        connection_str_os : str, optional, default: _AZURE_CONNECTION_STR_OS
            Key in the os environment for the Azure connection string.

        Returns
        -------
        AzureBlobDataHandler

        Raises
        ------
        KeyError
            When a os variable is not set.
        """

        connection_str_os = connection_str_os or _AZURE_CONNECTION_STR_OS

        client = client or _AZURE_CLIENT_NAME
        connection_str = os.environ[connection_str_os]
        return cls(client, connection_str)

    # --------------------------------------------------------------------------
    # Blob inspection

    def ls(self, path: str | Path | None = None) -> list[str]:

        # Provide slash from right
        if path is not None:
            path = f"{Path(path).as_posix()}/"

        # List all files and folders
        return [f.name for f in self._container.walk_blobs(path)]  # type: ignore

    def ls_files(self, path: str | Path | None = None) -> list[str]:

        return [f for f in self.ls(path) if not f.endswith("/")]

    def ls_folders(self, path: str | Path | None = None) -> list[str]:

        return [f for f in self.ls(path) if f.endswith("/")]

    # --------------------------------------------------------------------------
    # File handling

    def get_blob_client(self, path: str | Path) -> BlobClient:
        # Check input
        if isinstance(path, Path):
            path = str(path.as_posix())

        # Get blob client
        return self._container.get_blob_client(path)

    def get_metadata(self, path: str | Path) -> dict[str, str | float | int]:

        props = self.get_blob_client(path).get_blob_properties()
        return {  # type: ignore
            "file_name": Path(props.name).name,  # type: ignore
            "full_path": props.name,
            "container": props.container,
            # "creation_time": props.creation_time.timestamp(),
            "last_modified": props.last_modified.timestamp(),
        }

    def get_size(self, path: str | Path) -> int:

        return self.get_blob_client(path).get_blob_properties().size  # type: ignore

    def download_file(
        self, path: str | Path, local_path: str | Path, match_timestamps: bool = True
    ) -> None:

        blob = self.get_blob_client(path)

        # Ensure that local_path's parent exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(local_path, Path):
            local_path = str(local_path.as_posix())

        # Write blob data to local_path
        with open(local_path, "wb") as f:
            f.write(blob.download_blob().readall())

        # Manipulate file properties
        if match_timestamps:
            properties = blob.get_blob_properties()
            created: float = properties.creation_time.timestamp()
            last_modified: float = properties.last_modified.timestamp()
            os.utime(local_path, (created, last_modified))

    def read_json(self, path: str | Path) -> list | dict:

        blob = self.get_blob_client(path)
        return json.loads(blob.download_blob().readall())

    def read_pandas(
        self, path: str | Path, n_retries: int = 1, **kwargs
    ) -> pd.Series | pd.DataFrame:

        from amplo.utils.io import FILE_READERS

        # Check whether a proper file reader exists for the file
        file_extension = Path(path).suffix
        pandas_reader = FILE_READERS.get(file_extension)
        if pandas_reader is None:
            raise NotImplementedError(f"File format {file_extension} not supported.")

        # Read buffered data into pandas
        blob = self.get_blob_client(path)
        try:
            file_buffer = io.BytesIO(blob.download_blob().readall())
        except requests.exceptions.ConnectionError as err:
            if n_retries > 0:
                return self.read_pandas(path, n_retries - 1, **kwargs)
            raise err
        return pandas_reader(file_buffer, **kwargs)


@deprecated("This class won't be updated anymore.")
class AzureSynchronizer:
    def __init__(
        self,
        connection_string_name: str = "AZURE_STORAGE_STRING",
        container_client_name: str = "amploplatform",
        verbose=0,
    ):
        """
        Connector to Azure storage blob for downloading data that is stored
        in Amplo`s data storage fashion.

        Parameters
        ----------
        connection_string_name : str
        container_client_name : str
        verbose : int
        """
        client = BlobServiceClient.from_connection_string(
            os.getenv(connection_string_name, "unkown_azure_connection_string")
        )
        self.container = client.get_container_client(container_client_name)
        self.verbose = int(verbose)
        self._metadata_filename = ".metadata"
        self._str_time_format = "%Y-%m-%d %H:%M:%S:%z"

    def get_dir_paths(self, path=None):
        """
        Get all directories that are direct children of given directory (``path``).

        Parameters
        ----------
        path : str or Path, optional
            Path to search for directories.
            If not provided, searches in root `/`.

        Returns
        -------
        list of str
        """
        if path is not None:
            # Provide a slash from right
            path = f"{Path(path).as_posix()}/"
        dirs = [
            b.name for b in self.container.walk_blobs(path) if str(b.name).endswith("/")
        ]
        return dirs

    def get_filenames(self, path, with_prefix=False, sub_folders=False):
        """
        Get all files that are direct children of given directory (``path``).

        Parameters
        ----------
        path : str or Path
            Path to search for files
        with_prefix : bool
            Whether to fix the prefix of the files
        sub_folders : bool
            Whether to search also for files inside sub-folders

        Returns
        -------
        list of str
        """
        # Provide a slash from right
        path = f"{Path(path).as_posix()}/"

        # List files
        if sub_folders:
            files = [
                f.name
                for f in self.container.walk_blobs(path, delimiter="")
                if "." in f.name
            ]
        else:
            files = [
                f.name
                for f in self.container.walk_blobs(path, delimiter="")
                if f.name.count("/") == path.count("/") and "." in f.name
            ]

        # Fix prefix
        if not with_prefix:
            files = [f[len(path) :] for f in files]

        # Remove empties
        if "" in files:
            files.remove("")

        return files

    def sync_files(self, blob_dir, local_dir):
        """
        Download all files inside blob directory and store it to the local directory.

        Additionally, creates a file `.metadata` that stores additional info about
        synchronization such as blob directory path and last modification date.

        Parameters
        ----------
        blob_dir : str or Path
            Search directory (download)
        local_dir : str or Path
            Local directory (store)

        Returns
        -------
        found_new_data : bool
            Whether new data has been downloaded

        Notes
        -----
        The data in the `.metadata` file is currently only used for checking the last
        modification date, thus telling whether files have to be downloaded / updated.
        """
        # Set up paths
        blob_dir = Path(blob_dir)
        local_dir = Path(local_dir)
        metadata_dir = local_dir

        # Warn when names don't match
        if blob_dir.name != local_dir.name:
            warnings.warn(
                f"Name mismatch detected. {blob_dir.name} != {local_dir.name}"
            )

        # Skip "Random"
        if blob_dir.name == "Random":
            warnings.warn(f"Skipped synchronization from {blob_dir}")
            return False

        # Set up metadata
        blob_dir_properties = self.container.get_blob_client(
            str(blob_dir)
        ).get_blob_properties()
        metadata: dict[str, str | list[str] | datetime] = dict(
            blob_dir=str(blob_dir),
            container=blob_dir_properties.container,
            last_modified=blob_dir_properties.last_modified,
            new_files=[],
        )
        metadata["last_modified"] = cast(datetime, metadata["last_modified"])
        metadata["new_files"] = cast(list[str], metadata["new_files"])

        # Load local metadata from previous synchronization
        local_metadata = self.load_local_metadata(metadata_dir, not_exist_ok=True)
        last_updated = local_metadata.get(
            "last_modified", datetime(1900, 1, 1, tzinfo=pytz.UTC)
        )

        # Read & write all files
        for file in self.get_filenames(blob_dir):

            # Create directory only if files are found
            local_dir.mkdir(parents=True, exist_ok=True)

            # Get file blob
            blob = self.container.get_blob_client(str(blob_dir / file))
            blob_properties = blob.get_blob_properties()

            # Download and save if file is new or modified
            file_created: datetime = blob_properties.creation_time
            file_last_modified: datetime = blob_properties.last_modified
            if file_last_modified > last_updated:
                # Write file
                with open(str(local_dir / file), "wb") as f:
                    f.write(blob.download_blob().readall())
                # Match timestamps of local file with blob
                os.utime(
                    str(local_dir / file),
                    (file_created.timestamp(), file_last_modified.timestamp()),
                )
                # Increment
                metadata["new_files"].append(str(file))

        # Check whether found new data
        found_new_data = metadata["last_modified"] > last_updated
        if found_new_data:
            # Store metadata
            self._dump_local_metadata(metadata, metadata_dir)

        return found_new_data

    # --- Utilities ---

    def load_local_metadata(self, local_dir, *, not_exist_ok=False):
        metadata_path = Path(local_dir) / self._metadata_filename
        if metadata_path.exists():
            # Get and check local metadata
            metadata = json.load(open(str(metadata_path), "r"))
            assert isinstance(metadata, dict), f"Damaged metadata in {metadata_path}"
            # Convert string to datetime object
            metadata["last_modified"] = datetime.strptime(
                metadata["last_modified"], self._str_time_format
            )
            return metadata
        elif not_exist_ok:
            return dict()
        else:
            raise FileNotFoundError(f"File {metadata_path} does not exist")

    def _dump_local_metadata(self, metadata, local_dir):
        metadata_path = Path(local_dir) / self._metadata_filename
        # Check metadata keys
        assert {"blob_dir", "container", "new_files", "last_modified"}.issubset(
            metadata.keys()
        ), "Invalid metadata"
        # Make datetime object JSON serializable
        metadata["last_modified"] = datetime.strftime(
            metadata["last_modified"], self._str_time_format
        )
        # Dump
        json.dump(metadata, open(str(metadata_path), "w"))
