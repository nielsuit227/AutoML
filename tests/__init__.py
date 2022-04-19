import os
import shutil


__all__ = ['rmtree_automl']


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
