#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from azure.storage.blob import BlobServiceClient
from requests import exceptions
from typing_extensions import Self

from amplo.utils.util import check_dtypes

if TYPE_CHECKING:
    from azure.storage.blob import BlobClient, ContainerClient

__all__ = ["AzureBlobDataAPI"]


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
        bsc: BlobServiceClient = BlobServiceClient.from_connection_string(
            connection_str
        )
        self._container: ContainerClient = bsc.get_container_client(client)

    def __repr__(self):
        """
        Readable string representation of the class.
        """

        return f"{self.__class__.__name__}({self.client})"

    @classmethod
    def from_os_env(
        cls, client: str | None = None, connection_str_os: str | None = None
    ) -> Self:
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
        return [str(f.name) for f in self._container.walk_blobs(path)]

    def ls_files(self, path: str | Path | None = None) -> list[str]:

        return [f for f in self.ls(path) if not f.endswith("/")]

    def ls_folders(self, path: str | Path | None = None) -> list[str]:

        return [f for f in self.ls(path) if f.endswith("/")]

    # --------------------------------------------------------------------------
    # File handling

    def get_blob_client(self, path: str | Path) -> "BlobClient":
        # Check input
        if isinstance(path, Path):
            path = str(path.as_posix())

        # Get blob client
        return self._container.get_blob_client(path)

    def get_metadata(self, path: str | Path) -> dict[str, str | float | int]:

        props = self.get_blob_client(path).get_blob_properties()
        return {
            "file_name": Path(props.name).name if props.name else "Unknown",
            "full_path": str(props.name) if props.name else "Unknown",
            "container": props.container if props.container else "Unknown",
            # "creation_time": props.creation_time.timestamp(),
            "last_modified": props.last_modified.timestamp()
            if props.last_modified
            else "Unknown",
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
            created: float = (
                properties.creation_time if properties.creation_time else datetime.now()
            ).timestamp()
            last_modified: float = (
                properties.last_modified if properties.last_modified else datetime.now()
            ).timestamp()
            os.utime(local_path, (created, last_modified))

    def read_json(self, path: str | Path) -> Any:

        blob = self.get_blob_client(path)
        return json.loads(blob.download_blob().readall())

    def read_pandas(
        self, path: str | Path, n_retries: int = 1, **kwargs
    ) -> pd.Series | pd.DataFrame:

        # Read buffered data into pandas
        blob = self.get_blob_client(path)
        try:
            file_buffer = io.BytesIO(blob.download_blob().readall())
        except exceptions.ConnectionError as err:
            if n_retries > 0:
                return self.read_pandas(path, n_retries - 1, **kwargs)
            raise err
        return pd.read_parquet(file_buffer)
