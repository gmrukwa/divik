import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from divik.core import configurable, DivikResult


_SAVERS = set()


def saver(fn):
    _SAVERS.add(fn)


def save(model, destination, **kwargs):
    for save_fn in _SAVERS:
        save_fn(model, destination, **kwargs)


@saver
@configurable(whitelist=['enabled'])
def save_pickle(model, destination, enabled=True, **kwargs):
    if not enabled:
        return
    logging.info('Saving model pickle.')
    with open(os.path.join(destination, 'model.pkl'), 'wb') as pkl:
        pickle.dump(model, pkl)


@saver
def save_divik(model, destination, **kwargs):
    if not hasattr(model, 'result_'):
        return
    if not isinstance(model.result_, DivikResult):
        logging.info("Skipping DiviK details save. Cause: result is None")
        return
    from .divik import make_merged, save_merged
    logging.info('Saving DiviK details.')
    logging.info('Saving DivikResult pickle.')
    with open(os.path.join(destination, 'result.pkl'), 'wb') as pkl:
        pickle.dump(model.result_, pkl)
    logging.info("Saving DiviK partitions.")
    merged = make_merged(model.result_).astype(np.int64)
    assert merged.shape[0] == model.result_.clustering.labels_.size
    xy = kwargs.get('xy', None)
    save_merged(destination, merged, xy)


@saver
def save_summary(model, destination, **kwargs):
    if not hasattr(model, 'labels_') or \
            not hasattr(model, 'n_clusters_') or \
            not hasattr(model, 'depth_'):
        return
    logging.info("Saving JSON summary.")
    with open(os.path.join(destination, 'summary.json'), 'w') as smr:
        json.dump({
            "depth": model.depth_,
            "number_of_clusters": model.n_clusters_,
            "mean_cluster_size": model.labels_.size / float(model.n_clusters_)
        }, smr)


@saver
def save_labels(model, destination, **kwargs):
    if not hasattr(model, 'labels_'):
        return
    logging.info("Saving final partition.")
    np.save(os.path.join(destination, 'final_partition.npy'), model.labels_)
    np.savetxt(os.path.join(destination, 'final_partition.csv'), model.labels_,
               delimiter=', ', fmt='%i')


@saver
def save_centroids(model, destination, **kwargs):
    if not hasattr(model, 'centroids_'):
        return
    logging.info("Saving centroids.")
    np.save(os.path.join(destination, 'centroids.npy'), model.centroids_)
    np.savetxt(os.path.join(destination, 'centroids.csv'), model.centroids_,
               delimiter=', ')


@saver
def save_filters(model, destination, **kwargs):
    if not hasattr(model, 'filters_'):
        return
    logging.info("Saving filters.")
    np.save(os.path.join(destination, 'filters.npy'), model.filters_)
    np.savetxt(os.path.join(destination, 'filters.csv'), model.filters_,
               delimiter=', ', fmt='%i')


@saver
def save_cluster_paths(model, destination, **kwargs):
    if not hasattr(model, 'reverse_paths_'):
        return
    rev = ['_'.join(map(str, p)) for p in model.reverse_paths_]
    pd.DataFrame({
        'path': rev,
        'cluster_number': list(model.reverse_paths_.values())
    }).to_csv(os.path.join(destination, 'paths.csv'))
