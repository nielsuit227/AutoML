#  Copyright (c) 2022 by Amplo.

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import pytz
from azure.storage.blob import BlobServiceClient

__all__ = ["AzureSynchronizer"]


class AzureSynchronizer:
    def __init__(
        self,
        connection_string_name="AZURE_STORAGE_STRING",
        container_client_name="amploplatform",
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
            os.getenv(connection_string_name)
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
        metadata = dict(
            blob_dir=str(blob_dir),
            container=blob_dir_properties.container,
            last_modified=blob_dir_properties.last_modified,
            new_files=[],
        )

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
                metadata["new_files"] += [str(file)]

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
