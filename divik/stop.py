"""Stop condition for DiviK.

stop.py

Copyright 2019 Grzegorz Mrukwa

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from functools import partial
from multiprocessing import Pool
from typing import Tuple

import numpy as np

import divik.distance as dst
import divik.score as sc
import divik.utils as u


def minimal_size(data: u.Data, size: int = 2) -> bool:
    """Check if region is smaller than predefined size."""
    return data.shape[0] <= size


def _split_into_one(data: u.Data) -> Tuple[u.IntLabels, u.Centroids]:
    labels = np.zeros(shape=(data.shape[0],), dtype=int)
    centroids = np.mean(data, axis=0, keepdims=True)
    return labels, centroids


class combine:
    """Combine stop conditions to be checked together."""

    def __init__(self, *args: u.StopCondition):
        self._conditions = args

    def __call__(self, data: u.Data) -> bool:
        """Check if there is any precaution for segmentation."""
        return any(precaution(data) for precaution in self._conditions)


class Gap:
    """GAP statistic-based stop condition."""

    def __init__(self, distance: dst.DistanceMetric,
                 split_into_two: u.SegmentationMethod,
                 n_trials: int = 100, seed: int = 0, correction: bool=True,
                 pool: Pool = None):
        self._split_into_two = split_into_two
        self.correction = correction
        adjusted_gap = partial(sc.gap, distance=distance, seed=seed,
                               n_trials=n_trials, pool=pool)
        self._gap_of_two = partial(adjusted_gap, split=split_into_two)
        self._gap_of_one = partial(adjusted_gap, split=_split_into_one)

    def __call__(self, data: u.Data) -> bool:
        """Check if segmentation is significantly justified."""
        labels, centroids = self._split_into_two(data)
        split_likelihood, split_deviation = self._gap_of_two(
            data, labels, centroids, return_deviation=True)
        labels, centroids = _split_into_one(data)
        dont_split_likelihood = self._gap_of_one(data, labels, centroids)
        if self.correction:
            return split_likelihood + split_deviation < dont_split_likelihood
        else:
            return split_likelihood < dont_split_likelihood
