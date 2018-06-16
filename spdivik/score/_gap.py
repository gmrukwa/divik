from functools import partial
from multiprocessing import Pool
from typing import Tuple

import numpy as np
import pandas as pd

from spdivik.distance import DistanceMetric
from spdivik.types import Centroids, Data, IntLabels, SegmentationMethod
from spdivik.seeding import seeded


def _dispersion(data: Data, labels: IntLabels, centroids: Centroids,
                distance: DistanceMetric) -> float:
    clusters = pd.DataFrame(data).groupby(labels)
    return float(np.sum([
        np.sum(distance(centroids[np.newaxis, label], cluster_members.values))
        for label, cluster_members in clusters
    ]))


def _dispersion_of_random_sample(seed: int,
                                 shape: Tuple[int, int],
                                 minima: np.ndarray,
                                 ranges: np.ndarray,
                                 split: SegmentationMethod,
                                 distance: DistanceMetric) -> float:
    np.random.seed(seed)
    sample = np.random.random_sample(shape) * ranges + minima
    labels, centroids = split(sample)
    return _dispersion(sample, labels, centroids, distance)


@seeded(wrapped_requires_seed=True)
def gap(data: Data, labels: IntLabels, centroids: Centroids,
        distance: DistanceMetric, split: SegmentationMethod,
        seed: int=0, n_trials: int = 100, pool: Pool=None) -> float:
    minima = np.min(data, axis=0)
    ranges = np.max(data, axis=0) - minima
    compute_dispersion = partial(_dispersion_of_random_sample,
                                 shape=data.shape,
                                 minima=minima,
                                 ranges=ranges,
                                 split=split,
                                 distance=distance)
    if pool is None:
        dispersions = list(map(compute_dispersion, range(seed, seed + n_trials)))
    else:
        dispersions = pool.map(compute_dispersion, range(seed, seed + n_trials))
    reference = _dispersion(data, labels, centroids, distance)
    gap_value = np.log(np.mean(dispersions)) - np.log(reference)
    return gap_value
