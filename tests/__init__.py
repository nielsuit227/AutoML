import shutil
import os


def rmtree(folder='AutoML', must_exist=False):
    if must_exist and not os.path.exists(folder):
        raise FileNotFoundError(f'Directory {folder} does not exist')
    if os.path.exists(folder):
        shutil.rmtree(folder)
