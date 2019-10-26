#!/usr/bin/env python
import json
import logging
import os
import pickle
import typing
import numpy as np
import pandas as pd
import skimage.io as sio

from divik._cli._data_io import DIVIK_RESULT_FNAME
from divik import DiviK
import divik._summary as _smr
import divik._cli._utils as sc
import divik._utils as u


def _make_summary(result: typing.Optional[u.DivikResult]):
    n_clusters = _smr.total_number_of_clusters(result)
    if result is not None:
        mean_cluster_size = result.clustering.labels_.shape[0] / float(n_clusters)
    else:
        mean_cluster_size = -1
    return {
        "depth": _smr.depth(result),
        "number_of_clusters": n_clusters,
        "mean_cluster_size": mean_cluster_size
    }


def _make_merged(result: typing.Optional[u.DivikResult]) -> np.ndarray:
    depth = _smr.depth(result)
    return np.hstack(
        _smr.merged_partition(result, limit + 1).reshape(-1, 1)
        for limit in range(depth)
    )


def _save_merged(destination: str, merged: np.ndarray, xy: np.ndarray=None):
    np.savetxt(os.path.join(destination, 'partitions.csv'),
               merged, delimiter=', ', fmt='%i')
    np.save(os.path.join(destination, 'partitions.npy'), merged)
    if xy is not None:
        for level in range(merged.shape[1]):
            np.save(
                os.path.join(destination, 'partition-{0}.npy'.format(level)),
                merged[:, level]
            )
            visualization = u.visualize(merged[:, level], xy=xy)
            image_name = os.path.join(destination, 'partition-{0}.png'.format(level))
            sio.imsave(image_name, visualization)
    final_partition = merged[:, -1]
    np.save(os.path.join(destination, 'final_partition.npy'), final_partition)
    np.savetxt(os.path.join(destination, 'final_partition.csv'), final_partition,
               delimiter=', ', fmt='%i')


def save(data: u.Data, divik: DiviK, destination: str, xy: np.ndarray=None):
    logging.info("Saving result.")
    logging.info("Saving pickle.")
    with open(os.path.join(destination, DIVIK_RESULT_FNAME), 'wb') as pkl:
        pickle.dump(divik.result_, pkl)
    with open(os.path.join(destination, 'model.pkl'), 'wb') as pkl:
        pickle.dump(divik, pkl)
    logging.info("Saving JSON summary.")
    with open(os.path.join(destination, 'summary.json'), 'w') as smr:
        json.dump(_make_summary(divik.result_), smr)
    if divik.result_ is not None:
        logging.info("Saving partitions.")
        merged = _make_merged(divik.result_)
        assert merged.shape[0] == divik.result_.clustering.labels_.size
        _save_merged(destination, merged, xy)
        logging.info("Saving centroids.")
        centroids = pd.DataFrame(data).groupby(merged[:, -1]).mean().values
        np.save(os.path.join(destination, 'centroids.npy'), centroids)
        np.savetxt(os.path.join(destination, 'centroids.csv'), centroids,
                   delimiter=', ')
    else:
        logging.info("Skipping partition save. Cause: result is None")


def main():
    data, config, destination, xy = sc.initialize()
    logging.info('Workspace initialized.')
    logging.info('Scenario configuration: {0}'.format(config))
    divik = DiviK(**config)
    logging.info("Launching experiment.")
    try:
        divik.fit(data)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise
    save(data, divik, destination, xy)


if __name__ == '__main__':
    main()
