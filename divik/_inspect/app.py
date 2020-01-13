import os
import pickle
from typing import Optional

import dash
import numpy as np

from divik.core import DivikResult
from divik._cli._data_io import load_data


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

_divik_result = None
_divik_result_path = None
_xy = None


def unpickle(path: str) -> Optional[DivikResult]:
    with open(os.path.join(path, 'result.pkl'), 'rb') as result_file:
        return pickle.load(result_file)


def divik_result(path: str=None) -> Optional[DivikResult]:
    global _divik_result
    if _divik_result is None and path is not None:
        global _divik_result_path
        _divik_result_path = path
        _divik_result = unpickle(path)
    return _divik_result


def divik_result_path() -> str:
    global _divik_result_path
    return _divik_result_path


def xy(path: str=None) -> np.ndarray:
    global _xy
    if _xy is None and path is not None:
        _xy = load_data(path).astype(int)
    return _xy
