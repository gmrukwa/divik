"""Access to MATLAB legacy functionalities."""

from contextlib import contextmanager
import logging
import os
import platform
import warnings

import gamred_native as gn

import numpy as np


# noinspection SpellCheckingInspection
_MATLAB_SEARCH_PATHS = \
    "/usr/local/MATLAB/MATLAB_Runtime/v96/runtime/glnxa64:" + \
    "/usr/local/MATLAB/MATLAB_Runtime/v96/bin/glnxa64:" + \
    "/usr/local/MATLAB/MATLAB_Runtime/v96/sys/os/glnxa64:" + \
    "/usr/local/MATLAB/MATLAB_Runtime/v96/extern/bin/glnxa64:"


_local_system = platform.system()

if _local_system == 'Windows':
    # Must be here. Doesn't work as contextmanager.
    # If you think different increase counter of wasted hours: 4
    os.environ['PATH'] = os.environ['PATH'].lower()


_logger = logging.getLogger(__name__)


@contextmanager
def _matlab_paths():
    if _local_system == 'Linux':
        _logger.log(logging.INFO, 'Modifying LD_LIBRARY_PATH.')
        old_env = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = _MATLAB_SEARCH_PATHS + old_env
        _logger.log(logging.INFO, os.environ['LD_LIBRARY_PATH'])
    elif _local_system == 'Darwin':
        raise NotImplementedError('OSX hosts are not supported.')
    try:
        yield
    finally:
        if _local_system == 'Linux':
            _logger.log(logging.INFO, 'Restoring LD_LIBRARY_PATH.')
            # noinspection PyUnboundLocalVariable
            os.environ['LD_LIBRARY_PATH'] = old_env
            _logger.log(logging.INFO, os.environ['LD_LIBRARY_PATH'])


_engine = None


def _ensure_engine():
    global _engine
    if _engine is None:
        with _matlab_paths():
            import MatlabAlgorithms.MsiAlgorithms
            # noinspection PyPackageRequirements
            import matlab
        _engine = MatlabAlgorithms.MsiAlgorithms.initialize()
    return _engine


class MatlabError(Exception):
    """Thrown when legacy computations failed."""

    pass


def find_thresholds_mcr(values: np.ndarray, max_components: int = 10,
                        throw_on_engine_error: bool = False) -> np.ndarray:
    """Find candidate thresholds for decomposition of values by GMM.

    @param values: vector of values to decompose
    @param max_components: maximal number of components to decompose into
    @param throw_on_engine_error: if true, an exception will be raised if
    legacy code fails. Returns empty thresholds array otherwise.
    @return: array of candidate thresholds from crossings between GMM
    components
    """
    offset = np.min(values)
    values = values - offset
    with _matlab_paths():
        engine = _ensure_engine()
        # noinspection PyUnresolvedReferences
        import MatlabAlgorithms.MsiAlgorithms
        # noinspection PyPackageRequirements
        import matlab
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = matlab.double([[element] for element in values.ravel()])
        try:
            thresholds = engine.fetch_thresholds(values,
                                                 'MaxComponents',
                                                 float(max_components),
                                                 'DisableWarnings',
                                                 True,
                                                 nargout=1)
        except Exception as ex:
            if throw_on_engine_error:
                raise MatlabError() from ex
            else:
                return np.array([])
        return np.array(thresholds).ravel() + offset


def find_thresholds_native(values: np.ndarray, max_components: int = 10) \
        -> np.ndarray:
    """Find candidate thresholds for decomposition of values by GMM.

    Parameters
    ----------
    values:
        vector of values to decompose

    max_components:
        maximal number of components to decompose into

    Returns
    -------
    array of candidate thresholds from crossings between GMM components
    """
    if values.size <= max_components:
        max_components = values.size
    if values.size == 0:
        return np.array([])
    if max_components <= 0:
        raise ValueError("max_components must be positive")
    if values.ndim != 1:
        raise ValueError("values must be 1D array")
    values = np.ascontiguousarray(values)
    offset = np.min(values)
    if np.min(values) == np.max(values):
        return np.array([])
    return gn.find_thresholds(values - offset, max_components) + offset


find_thresholds = find_thresholds_mcr
