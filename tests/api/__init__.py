#  Copyright (c) 2022 by Amplo.

from pathlib import Path

import pytest

from tests import rmtree

__all__ = ["TestAPI"]


class TestAPI:
    sync_dir = Path("./test_dir")

    @pytest.fixture(autouse=True)
    def rmtree_sync_dir(self):
        rmtree(self.sync_dir)
        yield
        rmtree(self.sync_dir)
