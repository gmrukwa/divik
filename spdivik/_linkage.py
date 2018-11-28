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
import spdivik._scripting as scr
from spdivik.kmeans._scripting.parsers import assert_configured
import spdivik.types as ty
import spdivik.visualize as vis


LinkageMatrix = NewType('LinkageMatrix', np.ndarray)
LinkageBackend = Callable[[ty.Data], LinkageMatrix]
Dendrogram = NewType('Dendrogram', Dict)
DendrogramBackend = Callable[[LinkageMatrix], Dendrogram]
SaveFigureBackend = Callable[[str], None]
Experiment = NamedTuple('Experiment', [
    ('linkage', LinkageBackend),
    ('dendrogram', DendrogramBackend),
    ('save_figure', SaveFigureBackend)
])


def flatten_linkage(linkage_matrix: LinkageMatrix) -> ty.IntLabels:
    """Flatten partition of the dataset"""
    # default MATLAB behavior, seen on the dendrogram with default color
    # threshold
    cophenetic_distance_threshold = 0.7 * np.max(linkage_matrix[:, 2])
    partition = hcl.fcluster(linkage_matrix,
                             t=cophenetic_distance_threshold,
                             criterion='distance').astype(int)
    return partition


def compute_centroids(data: ty.Data, partition: ty.IntLabels) -> ty.Data:
    """Find centroids of flat clusters"""
    return pd.DataFrame(data).groupby(partition).mean().values


def build_experiment(config) -> Experiment:
    """Create experiment from configuration"""
    assert_configured(config, 'linkage')
    linkage_config = config['linkage']
    assert_configured(linkage_config, 'method')
    assert_configured(linkage_config, 'metric')
    assert_configured(linkage_config, 'optimal_ordering')
    linkage = partial(hcl.linkage, **linkage_config)

    assert_configured(config, 'dendrogram')
    dendrogram_config = config['dendrogram']
    assert_configured(dendrogram_config, 'truncate_mode')
    assert_configured(dendrogram_config, 'p')
    assert_configured(dendrogram_config, 'color_threshold')
    assert_configured(dendrogram_config, 'orientation')
    assert_configured(dendrogram_config, 'count_sort')
    assert_configured(dendrogram_config, 'distance_sort')
    assert_configured(dendrogram_config, 'show_leaf_counts')
    assert_configured(dendrogram_config, 'leaf_font_size')
    assert_configured(dendrogram_config, 'show_contracted')
    dendrogram = partial(hcl.dendrogram, **dendrogram_config)

    assert_configured(config, 'plot')
    plot_config = config['plot']
    assert_configured(plot_config, 'dpi')
    assert_configured(plot_config, 'facecolor')
    assert_configured(plot_config, 'edgecolor')
    assert_configured(plot_config, 'orientation')
    assert_configured(plot_config, 'transparent')
    assert_configured(plot_config, 'frameon')
    assert_configured(plot_config, 'bbox_inches')
    assert_configured(plot_config, 'pad_inches')
    save_figure = partial(plt.savefig, **plot_config)

    experiment = Experiment(linkage, dendrogram, save_figure)

    return experiment


def save(linkage: LinkageMatrix, dendrogram: Dendrogram,
         partition: ty.IntLabels, centroids: ty.Data,
         save_figure: SaveFigureBackend, destination: str, xy: np.ndarray=None):
    """Save results of experiment into dedicated location"""
    fname = partial(os.path.join, destination)
    logging.info('Saving linkage in numpy format.')
    np.save(fname('linkage.npy'), linkage)
    logging.info('Converting linkage to MATLAB format.')
    matlab_linkage = hcl.to_mlab_linkage(linkage)
    logging.info('Saving linkage in MATLAB format.')
    sio.savemat(fname('linkage.mat'), {'linkage': matlab_linkage})
    logging.info('Saving flat partition.')
    np.save(fname('partition.npy'), partition)
    np.savetxt(fname('partition.csv'), partition, fmt='%i', delimiter=', ')
    if xy is not None:
        logging.info('Generating visulization.')
        visualization = vis.visualize(partition, xy)
        imsave(fname('partition.png'), visualization)
    logging.info('Saving centroids.')
    np.save(fname('centroids.npy'), centroids)
    np.savetxt(fname('centroids.csv'), centroids, delimiter=', ')
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
    try:
        linkage = experiment.linkage(data)
        dendrogram = experiment.dendrogram(linkage)
        partition = flatten_linkage(linkage)
        centroids = compute_centroids(data, partition)
        save(linkage, dendrogram, partition, centroids,
             experiment.save_figure, destination, xy)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise


if __name__ == '__main__':
    main()
