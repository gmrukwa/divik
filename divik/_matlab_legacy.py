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

def find_gaussian_mixtures(x: np.ndarray, counts:np.ndarray, KS: int = 10) -> dict:
    """Find parameters for decomposition of values by GMM.

    Parameters
    ----------
    values:
        vector of values to decompose

    KS:
        number of Gaussian components

    Returns
    -------
    dictionary with arrays of weights, means, standard deviations and log likelihood of decomposition
    """
    if x.size <= KS:
        KS = x.size
    if x.size == 0:
        return {"weights": np.array([]), "mu": np.array([]), "sig": np.array([]), "TIC": np.nan, "l_lik": np.nan}
    if KS <= 0:
        raise ValueError("KS must be positive")
    if x.ndim != 1:
        raise ValueError("x must be 1D array")
    if counts.ndim != 1:
        raise ValueError("counts must be 1D array")
    x = np.ascontiguousarray(x)
    counts = np.ascontiguousarray(counts)
    if np.min(x) == np.max(x):
        return {"weights": np.array([]), "mu": np.array([]), "sig": np.array([]), "TIC": np.nan, "l_lik": np.nan}
    with sigsegv_handler():
        return gn.find_gaussian_mixtures(x, counts, KS)
