import os
import warnings
from pathlib import Path
from azure.storage.blob import BlobServiceClient


__all__ = ['AzureSynchronizer']


class AzureSynchronizer:

    def __init__(
            self,
            connection_string_name='AZURE_STORAGE_STRING',
            container_client_name='amploplatform',
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
        client = BlobServiceClient.from_connection_string(os.getenv(connection_string_name))
        self.container = client.get_container_client(container_client_name)
        # TODO: Use Amplo`s logging instead of a print function
        self.verbose = int(verbose)

    def get_dir_paths(self, path: str = None):
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
            path = f'{Path(path)}/'
        dirs = [b.name for b in self.container.walk_blobs(path)
                if str(b.name).endswith('/')]
        return dirs

    def get_filenames(self, path, with_prefix=False, sub_folders=False):
        """
        Get all files that are direct children of given directory (``path``).

        Parameters
        ----------
        path : str
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
        path = f'{Path(path)}/'

        # List files
        if sub_folders:
            files = [f.name for f in self.container.walk_blobs(path, delimiter='') if '.' in f.name]
        else:
            files = [f.name for f in self.container.walk_blobs(path, delimiter='') if
                     f.name.count('/') == path.count('/') and '.' in f.name]

        # Fix prefix
        if not with_prefix:
            files = [f[len(path):] for f in files]

        # Remove empties
        if '' in files:
            files.remove('')

        return files

    def sync_files(self, blob_dir, local_dir, **kwargs):
        """
        Download all files inside blob directory and store it to the local directory.

        Parameters
        ----------
        blob_dir : str or Path
            Search directory (download)
        local_dir : str or Path
            Local directory (store)
        kwargs : dict
            To manipulate `self.get_files()` function
        """

        blob_dir = Path(blob_dir)
        local_dir = Path(local_dir)

        if blob_dir.name != local_dir.name:
            warnings.warn(f'Name mismatch detected. {blob_dir.name} != {local_dir.name}')

        if blob_dir.name == 'Random':
            warnings.warn(f'Skipped synchronization from {blob_dir}')
            return

        # Read & write all files
        for file in self.get_filenames(str(blob_dir), **kwargs):
            # Create directory only if files are found
            local_dir.mkdir(parents=True, exist_ok=True)

            # Read file
            blob = self.container.get_blob_client(str(blob_dir / file))
            # Save file
            with open(str(local_dir / file), "wb") as f:
                f.write(blob.download_blob().readall())
            if self.verbose:
                print(str(blob_dir / file))
