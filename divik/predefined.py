"""Predefined scenarios for DiviK segmentation.

predefined.py

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
from typing import Callable, Optional

import tqdm

import divik.distance as dst
import divik.divik as dv
import divik.utils as u

Divik = Callable[[u.Data], Optional[u.DivikResult]]


def basic(gap_trials: int = 100,
          distance_percentile: float = 99.,
          iters_limit: int = 100,
          distance: str = None,
          minimal_size: int = 20,
          rejection_size: int = 0,
          minimal_features_percentage: float = .01,
          fast_kmeans_iters: int = 10,
          k_max: int = 10,
          random_seed: int = 0,
          normalize_rows: bool = False,
          use_logfilters: bool = False,
          n_jobs: int = 1,
          progress_reporter: tqdm.tqdm = None) -> Divik:
    """GAP limited DiviK with percentile initialization.

    @param gap_trials: number of random datasets used in GAP statistic
    computation. Increases precision and computational overhead.
    @param distance_percentile: percentile of distance used for selection of
    initial representatives. Must be contained in range [0, 100] inclusive.
    Higher may reveal more nuances, but reduce robustness.
    @param iters_limit: limit of k-means iterations
    @param distance: distance metric
    @param minimal_size: minimal size of accepted cluster
    @rejection_size: size under which split will be rejected
    @rejection_percentage: percentage of size under which split will be rejected
    @param minimal_features_percentage: minimal percent of features preserved
    @param fast_kmeans_iters: limit of iterations for stop condition check
    @param k_max: maximal number of clusters considered by k-means algorithm
    @param correction_of_gap: whether to compute GAP with correction
    @param normalize_rows: should be specified for correlation metric, sets
    row mean to 0 and norm to 1
    @param use_logfilters: filters based on logarithm of feature characteristic
    when True
    @param pool: pool for parallel processing. Recommended maxtasksperchild
    equal to number of cores.
    @param progress_reporter: tqdm-alike progress reporting object
    @return: adjusted DiviK pipeline
    """
    assert gap_trials > 0, gap_trials
    assert 0 <= distance_percentile <= 100, distance_percentile
    assert iters_limit > 0, iters_limit
    if distance is None:
        distance = dst.KnownMetric.correlation.value
    known_metrics = {metric.value: metric for metric in dst.KnownMetric}
    assert distance in known_metrics, \
        "Distance {0} unknown. Known distances: {1}".format(distance, known_metrics)
    assert 0 <= minimal_size, minimal_size
    assert 0 <= minimal_features_percentage <= 1, minimal_features_percentage
    assert fast_kmeans_iters > 0, fast_kmeans_iters
    divik = partial(dv.divik,
                    fast_kmeans_iters=fast_kmeans_iters,
                    k_max=k_max,
                    n_jobs=n_jobs,
                    distance=distance,
                    distance_percentile=distance_percentile,
                    iters_limit=iters_limit,
                    normalize_rows=normalize_rows,
                    rejection_size=rejection_size,
                    progress_reporter=progress_reporter,
                    random_seed=random_seed,
                    gap_trials=gap_trials,
                    min_features_percentage=minimal_features_percentage,
                    minimal_size=minimal_size,
                    use_logfilters=use_logfilters)
    return divik
