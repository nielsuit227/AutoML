import unittest
import shutil
from pathlib import Path


__all__ = ['APITestCase']


class APITestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sync_dir = Path('./test_dir')

    def tearDown(self):
        if self.sync_dir.exists():
            # Remove synchronized data
            shutil.rmtree(self.sync_dir)
