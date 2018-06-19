from contextlib import contextmanager
import os
import platform

import numpy as np

_MATLAB_SEARCH_PATHS = \
    ":/usr/local/MATLAB/MATLAB_Runtime/v91/runtime/glnxa64" + \
    ":/usr/local/MATLAB/MATLAB_Runtime/v91/bin/glnxa64" + \
    ":/usr/local/MATLAB/MATLAB_Runtime/v91/sys/os/glnxa64" + \
    ":/usr/local/MATLAB/MATLAB_Runtime/v91/sys/opengl/lib/glnxa64"

_local_system = platform.system()

if _local_system == 'Windows':
    # Must be here. Doesn't work as contextmanager.
    # If you think different increase counter of wasted hours: 4
    os.environ['PATH'] = os.environ['PATH'].lower()


@contextmanager
def _matlab_paths():
    if _local_system == 'Linux':
        old_env = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = old_env + _MATLAB_SEARCH_PATHS
    elif _local_system == 'Darwin':
        raise NotImplementedError('OSX hosts are not supported.')
    try:
        yield
    finally:
        if _local_system == 'Linux':
            os.environ['LD_LIBRARY_PATH'] = old_env


with _matlab_paths():
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


def find_thresholds(values: np.ndarray, max_components: int = 10,
                    throw_on_engine_error: bool = True) -> np.ndarray:
    with _matlab_paths():
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
