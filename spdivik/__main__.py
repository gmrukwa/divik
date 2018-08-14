#!/usr/bin/env python
import argparse as _agp
import json
import logging
import multiprocessing
import os
import pickle
import time
import typing
import numpy as np
import scipy.io as scio
from tqdm import tqdm
import spdivik.predefined as pred
import spdivik.summary as _smr
import spdivik.types as ty
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
    logging.info("Using spdivik, version " + __version__)


def load_config(config_path, destination: str):
    logging.info("Reading configuration file: " + config_path)
    with open(config_path) as configfile:
        config = json.load(configfile)
    logging.info('Backing up configuration.')
    with open(os.path.join(destination, 'config.json'), 'w') as configfile:
        json.dump(config, configfile)
    return config


def build_experiment(config) -> typing.Tuple[pred.Divik, tqdm]:
    scenario = config.pop('scenario', None)
    available = pred.scenarios.keys()
    assert scenario in available, \
        "Unknown scenario {0}, available: {1}".format(scenario, available)
    pool = multiprocessing.Pool(maxtasksperchild=4)
    progress_reporter = tqdm()
    return pred.scenarios[scenario](pool=pool,
                                    progress_reporter=progress_reporter,
                                    **config), progress_reporter


def _load_mat(path: str) -> np.ndarray:
    data = scio.loadmat(path)
    assert len(data) == 1, \
        'There should be a single variable inside MAT-file: ' + path
    return np.array(list(data.values())[0])


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


def _make_summary(result: typing.Optional[ty.DivikResult]):
    n_clusters = _smr.total_number_of_clusters(result)
    if result is not None:
        mean_cluster_size = result.partition.shape[0] / float(n_clusters)
    else:
        mean_cluster_size = -1
    return {
        "depth": _smr.depth(result),
        "number_of_clusters": n_clusters,
        "mean_cluster_size": mean_cluster_size
    }


def _make_merged(result: typing.Optional[ty.DivikResult]) -> np.ndarray:
    depth = _smr.depth(result)
    return np.hstack(
        _smr.merged_partition(result, limit + 1)
        for limit in range(depth)
    )


def save(result: typing.Optional[ty.DivikResult], destination: str):
    logging.info("Saving result.")
    logging.info("Saving pickle.")
    with open(os.path.join(destination, 'result.pkl'), 'wb') as pkl:
        pickle.dump(result, pkl)
    logging.info("Saving JSON summary.")
    with open(os.path.join(destination, 'summary.json'), 'w') as smr:
        json.dump(_make_summary(result), smr)
    if result is not None:
        logging.info("Saving partitions.")
        np.savetxt(os.path.join(destination, 'partitions.csv'),
                   _make_merged(result),
                   delimiter=', ')
    else:
        logging.info("Skipping partition save. Cause: result is None")


def main():
    arguments = parse_args()
    destination = prepare_destination(arguments.destination)
    setup_logger(destination)
    config = load_config(arguments.config, destination)
    experiment, progress = build_experiment(config)
    data = load_data(arguments.source)
    progress.total = data.shape[0]
    logging.info("Launching experiment.")
    try:
        result = experiment(data)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise
    finally:
        progress.close()
    save(result, destination)


if __name__ == '__main__':
    main()
