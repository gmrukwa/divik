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


def find_thresholds(values: np.ndarray, max_components: int=10) -> np.ndarray:
    engine = _ensure_engine()
    values = matlab.double([[element] for element in values.ravel()])
    thresholds = engine.fetch_thresholds(values,
                                         'MaxComponents', float(max_components),
                                         nargout=1)
    return np.array(thresholds).ravel()
