import argparse as _agp
from functools import partial
import json
import logging
import os
import sys
import time
from typing import Dict, Tuple

import h5py
import numpy as np
from scipy import io as scio
import tqdm

from spdivik import __version__
import spdivik.types as ty


def parse_args():
    parser = _agp.ArgumentParser()
    parser.add_argument('--source', '-s',
                        help='Source file with data. Should contain'
                             ' observations in rows and features in columns.',
                        action='store', dest='source', required=True)
    parser.add_argument('--destination', '-d',
                        help='Destination directory into which results will be'
                             ' saved.',
                        action='store', dest='destination', required=True)
    parser.add_argument('--config', '-c',
                        help='Configuration file for the experiment.',
                        action='store', dest='config', required=True)
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()


def prepare_destination(destination: str) -> str:
    datetime = time.strftime("%Y%m%d-%H%M%S")
    destination = os.path.join(destination, datetime)
    os.makedirs(destination)
    return destination


def setup_logger(destination: str, verbose: bool=False):
    log_destination = os.path.join(destination, 'logs.txt')
    if verbose:
        log_format = '%(asctime)s [%(levelname)s] %(filename)40s:%(lineno)3s' \
                     + ' - %(funcName)40s\t%(message)s'
        log_level = logging.DEBUG
    else:
        log_format = '%(asctime)s [%(levelname)s]\t%(message)s'
        log_level = logging.INFO
    handlers = [
        logging.StreamHandler(tqdm.tqdm),
        logging.FileHandler(filename=log_destination,
                            mode='a')
    ]
    del logging.root.handlers[:]
    logging.basicConfig(format=log_format, level=log_level, handlers=handlers)
    version_notice = "Using " + sys.argv[0] + \
                     " (spdivik, version " + __version__ + ")"
    logging.info(version_notice)


def load_config(config_path, destination: str):
    logging.info("Reading configuration file: " + config_path)
    with open(config_path) as configfile:
        config = json.load(configfile)
    logging.info('Backing up configuration.')
    with open(os.path.join(destination, 'config.json'), 'w') as configfile:
        json.dump(config, configfile)
    return config


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
        return _load_mat_with(path, backend=h5py.File, ignore='#')


def load_data(path: str) -> ty.Data:
    logging.info("Loading data: " + path)
    normalized = path.lower()
    if normalized.endswith('.csv'):
        loader = partial(np.loadtxt, delimiter=',')
    elif normalized.endswith('.txt'):
        loader = np.loadtxt
    elif normalized.endswith('.npy') or normalized.endswith('.npz'):
        loader = np.load
    elif normalized.endswith('.mat'):
        loader = _load_mat
    else:
        message = 'Unsupported data format: ' + os.path.splitext(path)[1]
        logging.error(message)
        raise IOError(message)
    return loader(path)


Config = Dict
DestinationPath = str


def initialize() -> Tuple[ty.Data, Config, DestinationPath]:
    arguments = parse_args()
    destination = prepare_destination(arguments.destination)
    setup_logger(destination, arguments.verbose)
    config = load_config(arguments.config, destination)
    try:
        data = load_data(arguments.source)
        logging.debug('Data loaded successfully.')
        return data, config, destination
    except Exception as ex:
        logging.error("Data loading failed with exception.")
        logging.error(repr(ex))
        raise
