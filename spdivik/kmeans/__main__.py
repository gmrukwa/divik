#!/usr/bin/env python
from functools import partial
import logging
from multiprocessing import Pool
import os
import pickle
from typing import Callable, NamedTuple, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import trange
import spdivik.kmeans as km
import spdivik.types as ty
import spdivik.score as sc
import spdivik._scripting as scr
import spdivik.kmeans._scripting.parsers as prs

Gap = Callable[
    [ty.Data, ty.IntLabels, ty.Centroids, ty.SegmentationMethod],
    float
]
Min, Max = int, int
Experiment = NamedTuple('Experiment', [
    ('kmeans', km.KMeans),
    ('gap', Gap),
    ('maximal_number_of_clusters', Max),
    ('minimal_number_of_clusters', Min),
    ('gap_pool', Pool),
    ('grouping_pool', Optional[Pool])
])


def build_experiment(config) -> Experiment:
    k_min, k_max = prs.parse_clustering_limits(config)
    distance = prs.parse_distance(config)
    labeling = km.Labeling(distance)
    initialization = prs.parse_initialization(config, distance)
    number_of_iterations = prs.parse_number_of_iterations(config)
    normalize_rows = prs.parse_row_normalization(config)
    kmeans = km.KMeans(labeling=labeling, initialize=initialization,
                       number_of_iterations=number_of_iterations,
                       normalize_rows=normalize_rows)

    gap_pool, grouping_pool = prs.spawn_pool(config)

    gap_trials = prs.parse_gap_trials(config)
    gap = partial(sc.gap, distance=distance, n_trials=gap_trials, pool=gap_pool,
                  return_deviation=True)
    return Experiment(kmeans=kmeans,
                      gap=gap,
                      maximal_number_of_clusters=k_max,
                      minimal_number_of_clusters=k_min,
                      gap_pool=gap_pool,
                      grouping_pool=grouping_pool)


def split_into_one(data: ty.Data) -> Tuple[ty.IntLabels, ty.Centroids]:
    return np.zeros((data.shape[0],), dtype=int), np.mean(data, axis=0, keepdims=True)


class DataBoundKmeans:
    def __init__(self, kmeans, data):
        self.kmeans, self.data = kmeans, data

    def __call__(self, number_of_clusters):
        return self.kmeans(data=self.data,
                           number_of_clusters=number_of_clusters)


def split(data: ty.Data, kmeans: km.KMeans, pool: Pool,
          minimal_number_of_clusters: int,
          maximal_number_of_clusters: int):
    data_bound_kmeans = DataBoundKmeans(kmeans=kmeans, data=data)
    clustering_start = max(minimal_number_of_clusters, 2)
    clustering_end = maximal_number_of_clusters + 1
    if pool is not None:
        logging.info('Segmenting data in parallel.')
        segmentations = pool.map(data_bound_kmeans,
                                 range(clustering_start, clustering_end))
    else:
        logging.info('Segmenting data in sequential.')
        clusters_counts = trange(clustering_start, clustering_end)
        segmentations = [
            data_bound_kmeans(number_of_clusters) for number_of_clusters
            in clusters_counts
        ]
    logging.info('Data segmented.')
    if minimal_number_of_clusters == 1:
        segmentations = [split_into_one(data)] + segmentations
    return segmentations


def build_splitter(kmeans, number_of_clusters):
    if number_of_clusters == 1:
        return split_into_one
    else:
        return partial(kmeans, number_of_clusters=number_of_clusters)


def score_splits(segmentations, data: ty.Data, kmeans: km.KMeans, gap,
                 minimal_number_of_clusters: int,
                 maximal_number_of_clusters: int):
    logging.info('Scoring splits with GAP statistic.')
    clusters_counts = trange(minimal_number_of_clusters, maximal_number_of_clusters+1)
    scores = [
        gap(data=data, labels=labels, centroids=centroids,
            split=build_splitter(kmeans, number_of_clusters))
        for number_of_clusters, (labels, centroids)
        in zip(clusters_counts, segmentations)
    ]
    logging.info('Scoring completed.')
    return scores


def make_segmentations_matrix(segmentations) -> np.ndarray:
    return np.hstack([s[0].reshape(-1, 1) for s in segmentations])


def make_scores_report(scores,
                       minimal_number_of_clusters: int,
                       maximal_number_of_clusters: int) -> pd.DataFrame:
    report = pd.DataFrame(scores, columns=['GAP', 's_k'])
    report['number_of_clusters'] = range(minimal_number_of_clusters,
                                         maximal_number_of_clusters + 1)
    is_suggested = report.GAP[:-1].values > (report.GAP[1:] + report.s_k[1:])
    report['suggested_number_of_clusters'] = list(is_suggested) + [False]
    return report


def save(segmentations, scores, destination: str, experiment: Experiment):
    logging.info("Saving result.")
    logging.info("Saving segmentations.")
    with open(os.path.join(destination, 'segmentations.pkl'), 'wb') as pkl:
        pickle.dump(segmentations, pkl)
    partitions = make_segmentations_matrix(segmentations)
    np.savetxt(os.path.join(destination, 'partitions.csv'), partitions,
               delimiter=', ', fmt='%i')
    for i in range(partitions.shape[1]):
        np.savetxt(os.path.join(destination, 'partitions.{0}.csv').format(i),
                   partitions[:, i].reshape(-1, 1), delimiter=', ', fmt='%i')
    logging.info("Saving scores.")
    report = make_scores_report(scores, experiment.minimal_number_of_clusters,
                                experiment.maximal_number_of_clusters)
    report.to_csv(os.path.join(destination, 'scores.csv'))


def main():
    data, config, destination, xy = scr.initialize()
    experiment = build_experiment(config)
    try:
        segmentations = split(data, experiment.kmeans, experiment.grouping_pool,
                              experiment.minimal_number_of_clusters,
                              experiment.maximal_number_of_clusters)
        scores = score_splits(segmentations, data, experiment.kmeans,
                              experiment.gap,
                              experiment.minimal_number_of_clusters,
                              experiment.maximal_number_of_clusters)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise
    finally:
        experiment.gap_pool.close()
    save(segmentations, scores, destination, experiment)


if __name__ == '__main__':
    main()
