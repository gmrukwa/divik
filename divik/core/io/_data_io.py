import logging
import os
from functools import partial

import h5py
import numpy as np
import pandas as pd
from scipy import io as scio

import divik.core as u


def _load_mat_with(path: str, backend=scio.loadmat, ignore="__") -> np.ndarray:
    data = backend(path)
    logging.debug("Data file opened successfully.")
    key = [key for key in list(data.keys()) if not key.startswith(ignore)]
    logging.debug("Found variables: {0}".format(key))
    if len(key) != 1:
        raise ValueError(
            "There should be a single variable inside MAT-file: "
            + path
            + "\nWere: "
            + str(key)
        )
    logging.debug("Selecting variable: {0}".format(key[0]))
    selected = data[key[0]]
    logging.debug("Loaded variable from file.")
    contignuous = np.array(selected, dtype=float)
    logging.debug("Converted to contignuous.")
    return contignuous


def _load_mat(path: str) -> np.ndarray:
    logging.debug("Loading MAT-file: " + path)
    try:
        logging.debug("Trying out legacy MAT-file loader.")
        return _load_mat_with(path, backend=scio.loadmat, ignore="__")
    except NotImplementedError:  # v7.3 MATLAB HDF5 MAT-File
        logging.debug("Legacy MAT-file loader failed, restarting with HDF5 loader.")
        hdf5 = partial(h5py.File, mode="r")
        return _load_mat_with(path, backend=hdf5, ignore="#").T


def load_data(path: str) -> u.Data:
    """Load 2D tabular data from file"""
    logging.info("Loading data: " + path)
    normalized = path.lower()
    if normalized.endswith(".csv"):
        loader = partial(np.loadtxt, delimiter=",")
    elif normalized.endswith(".txt"):
        loader = np.loadtxt
    elif normalized.endswith(".npy"):
        loader = np.load
    elif normalized.endswith(".mat"):
        loader = _load_mat
    else:
        message = "Unsupported data format: " + os.path.splitext(path)[1]
        logging.error(message)
        raise IOError(message)
    return loader(path)


def try_load_data(path):
    """Load 2D tabular data from file with logging"""
    try:
        data = load_data(path)
        logging.debug("Data loaded successfully.")
    except Exception as ex:
        logging.error("Data loading failed with an exception.")
        logging.error(repr(ex))
        raise
    return data


def try_load_xy(path):
    """Load integer spatial coordinates with logging from file"""
    if path is not None:
        try:
            xy = load_data(path).astype(int)
            logging.debug("Coordinates loaded successfully.")
        except Exception as ex:
            logging.error("Coordinates loading failed with an exception.")
            logging.error(repr(ex))
            raise
    else:
        xy = None
    return xy


def save_csv(array: np.ndarray, fname: str):
    """Save array to csv"""
    pd.DataFrame(array).to_csv(fname, header=False, index=False)
