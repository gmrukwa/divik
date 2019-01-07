from __future__ import division
from functools import partial
from multiprocessing import Pool
from operator import attrgetter
from typing import List, Optional, Tuple

from functional import pipe
import numpy as np
import pandas as pd
from sklearn.base import clone

from spdivik.distance import DistanceMetric
from spdivik.kmeans._core import KMeans, parse_distance
from spdivik.score._picker import Picker
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


# TODO: Reduce the number of parameters introducing single KMeans object
@seeded(wrapped_requires_seed=True)
def gap_(data: Data, labels: IntLabels, centroids: Centroids,
         distance: DistanceMetric, split: SegmentationMethod,
         seed: int=0, n_trials: int = 100, pool: Pool=None,
         return_deviation: bool = False) -> float:
    minima = np.min(data, axis=0)
    ranges = np.max(data, axis=0) - minima
    compute_dispersion = partial(_dispersion_of_random_sample,
                                 shape=data.shape,
                                 minima=minima,
                                 ranges=ranges,
                                 split=split,
                                 distance=distance)
    if pool is None:
        dispersions = [compute_dispersion(i) for i in range(seed, seed + n_trials)]
    else:
        dispersions = pool.map(compute_dispersion, range(seed, seed + n_trials))
    reference = _dispersion(data, labels, centroids, distance)
    log_dispersions = np.log(dispersions)
    gap_value = np.mean(log_dispersions) - np.log(reference)
    result = (gap_value, )
    if return_deviation:
        standard_deviation = np.sqrt(1 + 1 / n_trials) * np.std(log_dispersions)
        result += (standard_deviation,)
    return result


def _fast_kmeans(kmeans: KMeans, max_iter: int = 10) -> SegmentationMethod:
    new = clone(kmeans)
    new.max_iter = max_iter
    get_meta = attrgetter('labels_', 'cluster_centers_')
    return pipe(new.fit, get_meta)


class GapPicker(Picker):
    def __init__(self, max_iter: int = 10, seed: int = 0, n_trials: int = 10,
                 correction: bool=True):
        self.max_iter = max_iter
        self.seed = seed
        self.n_trials = n_trials
        self.correction = correction

    def score(self, data: Data, estimators: List[KMeans], pool: Pool=None) \
            -> np.ndarray:
        scores = [
            gap_(data=data,
                 labels=estimator.labels_,
                 centroids=estimator.cluster_centers_,
                 distance=parse_distance(estimator.distance),
                 split=_fast_kmeans(estimator, self.max_iter),
                 seed=self.seed,
                 n_trials=self.n_trials,
                 pool=pool,
                 return_deviation=True)
            for estimator in estimators
        ]
        return np.array(scores)

    def select(self, scores: np.ndarray) -> Optional[int]:
        GAP = scores[:, 0]
        s_k = scores[:, 1]
        if self.correction:
            is_suggested = GAP[:-1] > (GAP[1:] + s_k[1:])
            suggested_locations = list(np.nonzero(is_suggested))
        else:
            suggested_locations = [np.argmax(GAP)]
        return suggested_locations[0] if suggested_locations else None

    def report(self, estimators: List[KMeans], scores: np.ndarray) \
            -> pd.DataFrame:
        GAP = scores[:, 0]
        s_k = scores[:, 1]
        best = self.select(scores)
        suggested = np.zeros((len(estimators) - 1,), dtype=bool)
        if best is not None:
            suggested[best] = True
        suggested = list(suggested)
        suggested.append(None)
        return pd.DataFrame(
            data={
                'number_of_clusters': [e.n_clusters for e in estimators],
                'GAP': GAP,
                's_k': s_k,
                'suggested_number_of_clusters': suggested
            },
            columns=[
                'number_of_clusters',
                'GAP',
                's_k',
                'suggested_number_of_clusters'
            ]
        )
