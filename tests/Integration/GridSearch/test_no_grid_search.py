import os.path
import shutil

from Amplo import Pipeline
from . import GridSearchTestCase


# TODO: Import this function (`from tests import rmtree_automl`) a.s.a.p.
#   Unfortunately, the branch containing this piece of code it not yet merged.
def rmtree_automl(f):
    """Decorator that removes the directory `AutoML` before and after executing the function."""

    def rmtree_automl_wrapper(*args, **kwargs):
        if os.path.exists('AutoML'):
            shutil.rmtree('AutoML')
        out = f(*args, **kwargs)
        if os.path.exists('AutoML'):
            shutil.rmtree('AutoML')
        return out

    return rmtree_automl_wrapper


class TestNoGridSearch(GridSearchTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.default_pipe_args = dict(extract_features=False, sequence=False, plot_eda=False)

    @staticmethod
    @rmtree_automl
    def _test_pipe(data: tuple, **pipeline_args):
        # Fit pipeline
        pipeline = Pipeline(**pipeline_args)
        pipeline.fit(*data)
        # Check results
        results = pipeline.results
        assert any(pipeline.results.loc[:, 'type'] == 'Hyper parameter'), \
            'No hyper parameter results were found'

    def test_grid_search_type(self):
        for mode in ('regression', 'classification'):
            data = (self.rx, self.ry) if mode == 'regression' else (self.cx, self.cy)
            self._test_pipe(data, grid_search_type=None, **self.default_pipe_args)

    def test_grid_search_iterations(self):
        for mode in ('regression', 'classification'):
            data = (self.rx, self.ry) if mode == 'regression' else (self.cx, self.cy)
            self._test_pipe(data, grid_search_iterations=0, **self.default_pipe_args)
