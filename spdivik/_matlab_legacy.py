import os

import numpy as np

os.environ['PATH'] = os.environ['PATH'].lower()
import MatlabAlgorithms.MsiAlgorithms as msi
import matlab


_engine = None


def _ensure_engine():
    global _engine
    if _engine is None:
        _engine = msi.initialize()
    return _engine


class MatlabError(Exception):
    pass


def find_thresholds(values: np.ndarray, max_components: int=10,
                    throw_on_engine_error: bool=True) -> np.ndarray:
    engine = _ensure_engine()
    values = matlab.double([[element] for element in values.ravel()])
    try:
        thresholds = engine.fetch_thresholds(values,
                                             'MaxComponents',
                                             float(max_components),
                                             nargout=1)
    except Exception as ex:
        if throw_on_engine_error:
            raise MatlabError() from ex
        else:
            return np.array([])
    return np.array(thresholds).ravel()
