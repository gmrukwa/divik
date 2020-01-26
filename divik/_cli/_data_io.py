from itertools import chain
import glob
import logging
import os
from functools import partial
from typing import List

import h5py
import numpy as np
from scipy import io as scio

import divik.core as u


def _load_mat_with(path: str, backend=scio.loadmat, ignore='__') -> np.ndarray:
    data = backend(path)
    logging.debug('Data file opened successfully.')
    key = [key for key in list(data.keys()) if not key.startswith(ignore)]
    logging.debug('Found variables: {0}'.format(key))
    if len(key) != 1:
        raise ValueError('There should be a single variable inside MAT-file: '
                         + path + '\nWere: ' + str(key))
    logging.debug('Selecting variable: {0}'.format(key[0]))
    selected = data[key[0]]
    logging.debug('Loaded variable from file.')
    contignuous = np.array(selected, dtype=float)
    logging.debug('Converted to contignuous.')
    return contignuous


def _load_mat(path: str) -> np.ndarray:
    logging.debug('Loading MAT-file: ' + path)
    try:
        logging.debug('Trying out legacy MAT-file loader.')
        return _load_mat_with(path, backend=scio.loadmat, ignore='__')
    except NotImplementedError:  # v7.3 MATLAB HDF5 MAT-File
        logging.debug('Legacy MAT-file loader failed, restarting with HDF5 loader.')
        hdf5 = partial(h5py.File, mode='r')
        return _load_mat_with(path, backend=hdf5, ignore='#').T


def load_data(path: str) -> u.Data:
    logging.info("Loading data: " + path)
    normalized = path.lower()
    if normalized.endswith('.csv'):
        loader = partial(np.loadtxt, delimiter=',')
    elif normalized.endswith('.txt'):
        loader = np.loadtxt
    elif normalized.endswith('.npy'):
        loader = np.load
    elif normalized.endswith('.mat'):
        loader = _load_mat
    else:
        message = 'Unsupported data format: ' + os.path.splitext(path)[1]
        logging.error(message)
        raise IOError(message)
    return loader(path)


DIVIK_RESULT_FNAME = 'result.pkl'


def _result_path_patterns(slug: str) -> List[str]:
    slug_pattern = '*{0}*'.format(slug)
    direct = os.path.join(slug_pattern, DIVIK_RESULT_FNAME)
    prefixed = os.path.join('**', slug_pattern, DIVIK_RESULT_FNAME)
    suffixed = os.path.join(slug_pattern, '**', DIVIK_RESULT_FNAME)
    bothfixed = os.path.join('**', slug_pattern, '**', DIVIK_RESULT_FNAME)
    return list((direct, prefixed, suffixed, bothfixed))


def _find_possible_directories(patterns: List[str]) -> List[str]:
    possible_locations = chain.from_iterable(
        glob.glob(pattern, recursive=True) for pattern in patterns)
    possible_paths = list({
        os.path.split(fname)[0] for fname in possible_locations
    })
    return possible_paths


def as_divik_result_path(path_or_slug: str):
    possible_location = os.path.join(path_or_slug, DIVIK_RESULT_FNAME)
    if os.path.exists(possible_location):
        return path_or_slug
    patterns = _result_path_patterns(path_or_slug)
    possible_paths = _find_possible_directories(patterns)
    if not possible_paths:
        raise FileNotFoundError(path_or_slug)
    if len(possible_paths) > 1:
        msg = 'Multiple possible result directories: {0}. Selecting {1}.'
        logging.warning(msg.format(possible_paths, possible_paths[0]))
    return possible_paths[0]
