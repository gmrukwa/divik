#!/usr/bin/env python
from functools import partial
import logging
import os
import pickle
from typing import List, Tuple

import numpy as np
import skimage.io as sio

from divik.cluster import GAPSearch, KMeans
import divik._cli._utils as scr
import divik.core as u


Segmentations = List[Tuple[u.IntLabels, u.Centroids]]


def get_segmentations(kmeans: GAPSearch) -> Segmentations:
    return [(est.labels_, est.cluster_centers_) for est in kmeans.estimators_]


def make_segmentations_matrix(kmeans: GAPSearch) -> np.ndarray:
    return np.hstack([e.labels_.reshape(-1, 1) for e in kmeans.estimators_])


def save(kmeans: GAPSearch, destination: str, xy: np.ndarray=None):
    logging.info("Saving result.")

    logging.info("Saving model.")
    fname = partial(os.path.join, destination)
    with open(fname('model.pkl'), 'wb') as pkl:
        pickle.dump(kmeans, pkl)

    logging.info("Saving segmentations.")

    np.savetxt(fname('final_partition.csv'), kmeans.labels_.reshape(-1, 1),
               delimiter=', ', fmt='%i')
    np.save(fname('final_partition.npy'), kmeans.labels_.reshape(-1, 1))

    segmentations = get_segmentations(kmeans)
    with open(fname('segmentations.pkl'), 'wb') as pkl:
        pickle.dump(segmentations, pkl)

    partitions = make_segmentations_matrix(kmeans)
    np.savetxt(fname('partitions.csv'), partitions, delimiter=', ', fmt='%i')

    for i in range(partitions.shape[1]):
        np.savetxt(fname('partitions.{0}.csv').format(i + kmeans.min_clusters),
                   partitions[:, i].reshape(-1, 1), delimiter=', ', fmt='%i')

    if xy is not None:
        visualization = u.visualize(kmeans.labels_, xy=xy)
        sio.imsave(fname('partitions.{0}.png').format(kmeans.n_clusters_),
                   visualization)


def main():
    data, config, destination, xy = scr.initialize()
    try:
        single_kmeans = KMeans(**config['kmeans'])
        kmeans = GAPSearch(single_kmeans, **config['gap'])
        kmeans.fit(data)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise
    save(kmeans, destination, xy)


if __name__ == '__main__':
    main()
