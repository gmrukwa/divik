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
    return parser.parse_args()


def prepare_destination(destination: str) -> str:
    datetime = time.strftime("%Y%m%d-%H%M%S")
    destination = os.path.join(destination, datetime)
    os.makedirs(destination)
    return destination


def setup_logger(destination: str):
    log_destination = os.path.join(destination, 'logs.txt')
    log_format = '%(asctime)s [%(levelname)s] %(filename)40s:%(lineno)3s' \
                 + ' - %(funcName)40s\t%(message)s'
    logging.basicConfig(format=log_format, level=logging.DEBUG, filemode='w',
                        filename=log_destination)
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
    key = [key for key in data.keys() if not key.startswith(ignore)]
    assert len(key) == 1, \
        'There should be a single variable inside MAT-file: ' + path \
        + '\nWere: ' + str(key)
    return np.array(data[key[0]])


def _load_mat(path: str) -> np.ndarray:
    try:
        return _load_mat_with(path, backend=scio.loadmat, ignore='__')
    except NotImplementedError:  # v7.3 MATLAB HDF5 MAT-File
        return _load_mat_with(path, backend=h5py.File, ignore='#')


def load_data(path: str) -> ty.Data:
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


Config = Dict
DestinationPath = str


def initialize() -> Tuple[ty.Data, Config, DestinationPath]:
    arguments = parse_args()
    destination = prepare_destination(arguments.destination)
    setup_logger(destination)
    config = load_config(arguments.config, destination)
    data = load_data(arguments.source)
    return data, config, destination
