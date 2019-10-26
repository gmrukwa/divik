import argparse as _agp
import json
import logging
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tqdm

import divik._utils as u
from divik import __version__
from divik._cli._data_io import load_data


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
    parser.add_argument('--xy',
                        help='File with spatial coordinates X and Y. Should '
                             'contain X in first column, Y in second. The '
                             'number of rows should be equal to the data.',
                        action='store', dest='xy', required=False, default=None)
    parser.add_argument('--omit-datetime', action='store_true',
                        help='If defines, omit the datetime subdirectory of '
                        'result.')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()


def prepare_destination(destination: str, omit_datetime: bool = False) -> str:
    if not omit_datetime:
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
                     " (divik, version " + __version__ + ")"
    logging.info(version_notice)


def load_config(config_path, destination: str):
    logging.info("Reading configuration file: " + config_path)
    with open(config_path) as configfile:
        config = json.load(configfile)
    logging.info('Backing up configuration.')
    with open(os.path.join(destination, 'config.json'), 'w') as configfile:
        json.dump(config, configfile)
    return config


Config = Dict
DestinationPath = str
Coordinates = np.ndarray


def try_load_data(path):
    try:
        data = load_data(path)
        logging.debug('Data loaded successfully.')
    except Exception as ex:
        logging.error("Data loading failed with an exception.")
        logging.error(repr(ex))
        raise
    return data


def try_load_xy(path):
    if path is not None:
        try:
            xy = load_data(path).astype(int)
            logging.debug('Coordinates loaded successfully.')
        except Exception as ex:
            logging.error('Coordinates loading failed with an exception.')
            logging.error(repr(ex))
            raise
    else:
        xy = None
    return xy


def initialize() -> Tuple[u.Data, Config, DestinationPath, Coordinates]:
    arguments = parse_args()
    destination = prepare_destination(arguments.destination,
                                      arguments.omit_datetime)
    setup_logger(destination, arguments.verbose)
    config = load_config(arguments.config, destination)
    data = try_load_data(arguments.source)
    xy = try_load_xy(arguments.xy)
    return data, config, destination, xy


def save_csv(array: np.ndarray, fname: str):
    pd.DataFrame(array).to_csv(fname, header=False, index=False)
