"""Access to MATLAB legacy functionalities."""
import faulthandler
from contextlib import contextmanager

import numpy as np

import gamred_native as gn


@contextmanager
def sigsegv_handler():
    was_enabled = faulthandler.is_enabled()
    faulthandler.enable()
    yield
    if not was_enabled:
        faulthandler.disable()


def find_thresholds(values: np.ndarray, max_components: int = 10) -> np.ndarray:
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
    with sigsegv_handler():
        return gn.find_thresholds(values - offset, max_components) + offset
