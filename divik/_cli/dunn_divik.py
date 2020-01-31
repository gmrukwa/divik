#!/usr/bin/env python
import logging

from divik.core import build
from divik.cluster import DiviK, DunnSearch, GAPSearch, KMeans
import divik._cli._utils as sc

from divik._cli.divik import save


def _fast_kmeans(**config):
    distance = config.get('distance', 'correlation')
    normalize_rows = config.get('normalize_rows', None)
    if normalize_rows is None:
        normalize_rows = distance == 'correlation'
    single_kmeans = KMeans(
        n_clusters=2,
        distance=distance,
        init='percentile',
        percentile=config.get('distance_percentile', 99.0),
        max_iter=config.get('max_iter', 10),
        normalize_rows=normalize_rows,
    )
    kmeans = GAPSearch(
        single_kmeans,
        max_clusters=2,
        n_jobs=config.get('n_jobs', None),
        seed=config.get('random_seed', 0),
        n_trials=config.get('gap_trials', 10),
        sample_size=config.get('sample_size', 10000),
        verbose=config.get('verbose', False),
    )
    return kmeans


def _full_kmeans(**config):
    distance = config.get('distance', 'correlation')
    normalize_rows = config.get('normalize_rows', None)
    single_kmeans = KMeans(
        n_clusters=2,
        distance=distance,
        init='percentile',
        percentile=config.get('distance_percentile', 99.0),
        max_iter=config.get('max_iter', 100),
        normalize_rows=normalize_rows,
    )
    kmeans = DunnSearch(
        single_kmeans,
        max_clusters=config.get('k_max', 10),
        n_jobs=config.get('n_jobs', None),
        verbose=config.get('verbose', False),
    )
    return kmeans


def main():
    data, config, destination, xy = sc.initialize()
    logging.info('Workspace initialized.')
    logging.info('Scenario configuration: {0}'.format(config))
    fast = _fast_kmeans(**config)
    full = _full_kmeans(**config)
    divik_config = {'kmeans': full, 'fast_kmeans': fast, **config}
    divik = build(DiviK, **divik_config)
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
