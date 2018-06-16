from functools import partial
from multiprocessing import Pool
from typing import Tuple

import numpy as np

import spdivik.distance as dst
import spdivik.types as ty
import spdivik.score as sc


def minimal_size(data: ty.Data, size: int=2) -> bool:
    return data.shape[0] <= size


def _split_into_one(data: ty.Data) -> Tuple[ty.IntLabels, ty.Centroids]:
    labels = np.zeros(shape=(data.shape[0],), dtype=int)
    centroids = np.mean(data, axis=0, keepdims=True)
    return labels, centroids


class combine:
    def __init__(self, *args: ty.StopCondition):
        self._conditions = args

    def __call__(self, data: ty.Data) -> bool:
        return any(precaution(data) for precaution in self._conditions)


class Gap:
    def __init__(self, distance: dst.DistanceMetric,
                 split_into_two: ty.SegmentationMethod,
                 n_trials: int=100, seed: int=0, pool: Pool=None):
        self._split_into_two = split_into_two
        adjusted_gap = partial(sc.gap, distance=distance, seed=seed,
                               n_trials=n_trials, pool=pool)
        self._gap_of_two = partial(adjusted_gap, split=split_into_two)
        self._gap_of_one = partial(adjusted_gap, split=_split_into_one)

    def __call__(self, data: ty.Data) -> bool:
        labels, centroids = self._split_into_two(data)
        split_likelihood = self._gap_of_two(data, labels, centroids)
        labels, centroids = _split_into_one(data)
        dont_split_likelihood = self._gap_of_one(data, labels, centroids)
        return split_likelihood < dont_split_likelihood
