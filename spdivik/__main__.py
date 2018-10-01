#!/usr/bin/env python
import json
import logging
import multiprocessing
import os
import pickle
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm
import spdivik.predefined as pred
import spdivik.summary as _smr
import spdivik.types as ty
import spdivik._scripting as sc


def build_experiment(config, data: np.ndarray) -> typing.Tuple[pred.Divik, tqdm]:
    scenario = config.pop('scenario', None)
    available = pred.scenarios.keys()
    assert scenario in available, \
        "Unknown scenario {0}, available: {1}".format(scenario, available)
    pool = multiprocessing.Pool(**config.pop('pool', {
        'maxtasksperchild': 4
    }))
    if 'minimal_size_percentage' in config:
        config['minimal_size'] = int(data.shape[0] * config.pop('minimal_size_percentage', 0.01))
    progress_reporter = tqdm()
    logging.info('Scenario configuration: {0}'.format(config))
    return pred.scenarios[scenario](pool=pool,
                                    progress_reporter=progress_reporter,
                                    **config), progress_reporter


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
        _smr.merged_partition(result, limit + 1).reshape(-1, 1)
        for limit in range(depth)
    )


def save(data: ty.Data, result: typing.Optional[ty.DivikResult], destination: str):
    logging.info("Saving result.")
    logging.info("Saving pickle.")
    with open(os.path.join(destination, 'result.pkl'), 'wb') as pkl:
        pickle.dump(result, pkl)
    logging.info("Saving JSON summary.")
    with open(os.path.join(destination, 'summary.json'), 'w') as smr:
        json.dump(_make_summary(result), smr)
    if result is not None:
        logging.info("Saving partitions.")
        merged = _make_merged(result)
        np.savetxt(os.path.join(destination, 'partitions.csv'),
                   merged, delimiter=', ', fmt='%i')
        final_partition = merged[:, -1]
        np.save(os.path.join(destination, 'final_partition.npy'), final_partition)
        np.savetxt(os.path.join(destination, 'final_partition.csv'), final_partition,
                   delimiter=', ', fmt='%i')
        logging.info("Saving centroids.")
        centroids = pd.DataFrame(data).groupby(final_partition).mean().values
        np.save(os.path.join(destination, 'centroids.npy'), centroids)
        np.savetxt(os.path.join(destination, 'centroids.csv'), centroids,
                   delimiter=', ')
    else:
        logging.info("Skipping partition save. Cause: result is None")


def main():
    data, config, destination = sc.initialize()
    experiment, progress = build_experiment(config, data)
    progress.total = data.shape[0]
    progress.update(0)
    logging.info("Launching experiment.")
    try:
        result = experiment(data)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise
    finally:
        progress.close()
    save(data, result, destination)


if __name__ == '__main__':
    main()
