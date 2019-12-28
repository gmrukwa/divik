"""Access to MATLAB legacy functionalities."""
import gamred_native as gn
import numpy as np


def find_thresholds(values: np.ndarray, max_components: int = 10) \
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
