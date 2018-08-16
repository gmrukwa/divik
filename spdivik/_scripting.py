import argparse as _agp
import json
import logging
import os
import sys
import time

import numpy as np
from scipy import io as scio

from spdivik import __version__


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


def _load_mat(path: str) -> np.ndarray:
    data = scio.loadmat(path)
    assert len(data) == 2, \
        'There should be a single variable inside MAT-file: ' + path
    key = [key for key in data.keys() if not key.startswith('__')]
    return np.array(data[key])


def load_data(path: str) -> np.ndarray:
    logging.info("Loading data: " + path)
    path = path.lower()
    if path.endswith('.csv') or path.endswith('.txt'):
        loader = np.loadtxt
    elif path.endswith('.npy'):
        loader = np.load
    elif path.endswith('.mat'):
        loader = _load_mat
    else:
        message = 'Unsupported data format: ' + os.path.splitext(path)[1]
        logging.error(message)
        raise IOError(message)
    return loader(path)
