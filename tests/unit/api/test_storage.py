#  Copyright (c) 2022 by Amplo.

from amplo.api.storage import AzureSynchronizer
from tests.unit.api import TestAPI


class TestStorage(TestAPI):

    def test_get_dir_paths(self):
        blob_dir = 'Demo/Charger 75kW/data'
        sync = AzureSynchronizer()
        dirs_actual = sync.get_dir_paths(blob_dir)
        dirs_target = [f'{blob_dir}/{folder}/' for folder in ('Diagnostics', 'Healthy', 'Unlabelled')]
        assert set(dirs_actual) == set(dirs_target), \
            'The function\'s output did not match the expected output'

    def test_get_filenames(self):
        blob_dir = 'Demo/Charger 75kW/data/Diagnostics/xx'
        sync = AzureSynchronizer()
        files_actual = sync.get_filenames(blob_dir)
        files_target = ['beth10.csv', 'long-data.csv', 'long-dataxx.csv', 'long-dataxxy.csv']
        assert set(files_actual) == set(files_target), \
            'The function\'s output did not match the expected output'

    def test_sync_files(self):
        blob_dir = 'Demo/Charger 75kW/data/Diagnostics/xx'
        local_dir = self.sync_dir / 'Demo/Charger 75kW/Diagnostics/data/xx'
        # Set up API and synch data
        sync = AzureSynchronizer()
        sync.sync_files(blob_dir, local_dir)
        # Assert that files have been synchronized
        blob_files = sync.get_filenames(blob_dir)
        local_files = [path.name for path in local_dir.glob('*')]
        assert set(blob_files).issubset(local_files), 'Synchronization has failed. Content is not equal...'
