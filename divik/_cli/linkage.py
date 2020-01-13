"""Hierarchical clustering of the data"""
from functools import partial
import logging
import os
import pickle
from typing import Callable, Dict, NamedTuple, NewType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hcl
import scipy.io as sio
from skimage.io import imsave
import divik._cli._utils as scr
import divik.core as u


LinkageMatrix = NewType('LinkageMatrix', np.ndarray)
LinkageBackend = Callable[[u.Data], LinkageMatrix]
Dendrogram = NewType('Dendrogram', Dict)
DendrogramBackend = Callable[[LinkageMatrix], Dendrogram]
SaveFigureBackend = Callable[[str], None]
Experiment = NamedTuple('Experiment', [
    ('linkage', LinkageBackend),
    ('dendrogram', DendrogramBackend),
    ('save_figure', SaveFigureBackend)
])


def flatten_linkage(linkage_matrix: LinkageMatrix) -> u.IntLabels:
    """Flatten partition of the dataset"""
    # default MATLAB behavior, seen on the dendrogram with default color
    # threshold
    cophenetic_distance_threshold = 0.7 * np.max(linkage_matrix[:, 2])
    partition = hcl.fcluster(linkage_matrix,
                             t=cophenetic_distance_threshold,
                             criterion='distance').astype(int)
    return partition


def compute_centroids(data: u.Data, partition: u.IntLabels) -> u.Data:
    """Find centroids of flat clusters"""
    return pd.DataFrame(data).groupby(partition).mean().values


def assert_configured(config, name):
    assert name in config, 'Missing "' + name + '" field in config.'


def build_experiment(config) -> Experiment:
    """Create experiment from configuration"""
    assert_configured(config, 'linkage')
    linkage_config = config['linkage']
    for item in ['method', 'metric', 'optimal_ordering']:
        assert_configured(linkage_config, item)
    linkage = partial(hcl.linkage, **linkage_config)

    assert_configured(config, 'dendrogram')
    dendrogram_config = config['dendrogram']
    for item in ['truncate_mode', 'p', 'color_threshold', 'orientation',
                 'count_sort', 'distance_sort', 'show_leaf_counts',
                 'leaf_font_size', 'show_contracted']:
        assert_configured(dendrogram_config, item)
    dendrogram = partial(hcl.dendrogram, **dendrogram_config)

    assert_configured(config, 'plot')
    plot_config = config['plot']
    for item in ['dpi', 'facecolor', 'edgecolor', 'orientation', 'transparent',
                 'frameon', 'bbox_inches', 'pad_inches']:
        assert_configured(plot_config, item)
    save_figure = partial(plt.savefig, **plot_config)

    experiment = Experiment(linkage, dendrogram, save_figure)

    return experiment


def save_linkage(fname, linkage: LinkageMatrix):
    logging.info('Saving linkage in numpy format.')
    np.save(fname('linkage.npy'), linkage)
    logging.info('Converting linkage to MATLAB format.')
    matlab_linkage = hcl.to_mlab_linkage(linkage)
    logging.info('Saving linkage in MATLAB format.')
    sio.savemat(fname('linkage.mat'), {'linkage': matlab_linkage})


def save_partition(fname, partition: u.IntLabels, xy: np.ndarray=None):
    logging.info('Saving flat partition.')
    np.save(fname('partition.npy'), partition)
    np.savetxt(fname('partition.csv'), partition, fmt='%i', delimiter=', ')
    if xy is not None:
        logging.info('Generating visulization.')
        visualization = u.visualize(partition, xy)
        imsave(fname('partition.png'), visualization)


def save_centroids(fname, centroids: np.ndarray):
    logging.info('Saving centroids.')
    np.save(fname('centroids.npy'), centroids)
    np.savetxt(fname('centroids.csv'), centroids, delimiter=', ')


def save_dendrogram(fname, save_figure: SaveFigureBackend, dendrogram: Dendrogram):
    logging.info('Pickling dendrogram data.')
    with open(fname('dendrogram.pkl'), 'wb') as file:
        pickle.dump(dendrogram, file)
    logging.info('Saving dendrogram plot as PDF.')
    save_figure(fname('dendrogram.pdf'))
    logging.info('Saving dendrogram plot as PNG.')
    save_figure(fname('dendrogram.png'))
    plt.close('all')


def main():
    """Entry point of the script"""
    data, config, destination, xy = scr.initialize()
    experiment = build_experiment(config)
    fname = partial(os.path.join, destination)
    try:
        linkage = experiment.linkage(data)
        save_linkage(fname, linkage)
        dendrogram = experiment.dendrogram(linkage)
        save_dendrogram(fname, experiment.save_figure, dendrogram)
        partition = flatten_linkage(linkage)
        save_partition(fname, partition, xy)
        centroids = compute_centroids(data, partition)
        save_centroids(fname, centroids)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise


if __name__ == '__main__':
    main()
