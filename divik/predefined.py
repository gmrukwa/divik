"""Predefined scenarios for DiviK segmentation.

predefined.py

Copyright 2018 Spectre Team

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
from typing import Callable, Optional

import tqdm

import divik.distance as dst
import divik.divik as dv
import divik.feature_selection as fs
import divik.kmeans as km
import divik.rejection as rj
import divik.score as sc
import divik.stop as st
import divik.types as ty

Divik = Callable[[ty.Data], Optional[ty.DivikResult]]

scenarios = {}


def _scenario(f):
    scenarios[f.__name__] = f
    return f


def _dunn_optimized_kmeans(distance: dst.DistanceMetric,
                           kmeans: km._KMeans,
                           pool: Pool = None,
                           k_max: int = 10) -> sc.Optimizer:
    dunn = partial(sc.dunn, distance=distance)
    sweep_clusters_number = [
        sc.ParameterValues('number_of_clusters', list(range(2, k_max + 1)))
    ]
    best_kmeans_with_dunn = sc.Optimizer(score=dunn,
                                         segmentation_method=kmeans,
                                         parameters=sweep_clusters_number,
                                         pool=pool)
    return best_kmeans_with_dunn


_AMPLITUDE_FILTER = fs.FilteringMethod(
    'amplitude',
    partial(fs.select_by,
            statistic=fs.amplitude,
            discard_up_to=1,
            # discarding only lowest component, if possible
            preserve_topmost=True))
_VARIANCE_FILTER = fs.FilteringMethod(
    'variance',
    partial(fs.select_by,
            statistic=fs.variance,
            # selecting only most varying component, if possible
            discard_up_to=-1,
            preserve_topmost=True))
_LOG_AMPLITUDE_FILTER = fs.FilteringMethod(
    'log_amplitude',
    partial(fs.select_by,
            statistic=fs.log_amplitude,
            discard_up_to=1,
            # discarding only lowest component, if possible
            preserve_topmost=True))
_LOG_VARIANCE_FILTER = fs.FilteringMethod(
    'log_variance',
    partial(fs.select_by,
            statistic=fs.log_variance,
            # selecting only most varying component, if possible
            discard_up_to=-1,
            preserve_topmost=True))


class _PrefilteringWrapper:
    def __init__(self, prefilter: fs.FilteringMethod, divik: Divik):
        self._prefilter = prefilter
        self._divik = divik

    def __call__(self, data: ty.Data) -> ty.DivikResult:
        preselection, threshold = self._prefilter(data)
        result = self._divik(data[:, preselection])
        result.thresholds[self._prefilter.name] = threshold
        result.filters[self._prefilter.name] = preselection
        return result


@_scenario
def prefiltered_correlative(minimal_split_segment: int = 20, iters_limit: int = 100,
                            progress_reporter: tqdm.tqdm = None, pool: Pool = None) -> Divik:
    """Size limited DiviK with extreme initialization and correlation.

    Uses feature abundance pre-filtering before segmentation.

    DiviK preset as in: P. Widlak, G. Mrukwa, M. Kalinowska, M. Pietrowska,
    M. Chekan, J. Wierzgon, M. Gawin, G. Drazek and J. Polanska, "Detection of
    molecular signatures of oral squamous cell carcinoma and normal epithelium
    - application of a novel methodology for unsupervised segmentation of
    imaging mass spectrometry data," Proteomics, vol. 16, no. 11-12,
    pp. 1613-21, 2016

    @param minimal_split_segment: lowest size of region to split
    @param iters_limit: limit of k-means iterations
    @param progress_reporter: tqdm-alike progress reporting object
    @param pool: pool for parallel processing. Recommended maxtasksperchild
    equal to number of cores.
    @return: adjusted DiviK pipeline
    """
    distance = dst.ScipyDistance(dst.KnownMetric.correlation)
    kmeans = km._KMeans(labeling=km.Labeling(distance),
                        initialize=km.ExtremeInitialization(distance),
                        number_of_iterations=iters_limit)
    best_kmeans_with_dunn = _dunn_optimized_kmeans(
        distance, kmeans, pool=pool)
    stop_for_small_size = partial(st.minimal_size, size=minimal_split_segment)
    divik = partial(dv.divik,
                    split=best_kmeans_with_dunn,
                    feature_selectors=[_VARIANCE_FILTER],
                    stop_condition=stop_for_small_size,
                    progress_reporter=progress_reporter,
                    min_features_percentage=.05)
    prefiltered_divik = _PrefilteringWrapper(prefilter=_AMPLITUDE_FILTER,
                                             divik=divik)
    return prefiltered_divik


@_scenario
def master(gap_trials: int = 100, distance_percentile: float = 99.,
           iters_limit: int = 100, pool: Pool = None,
           progress_reporter: tqdm.tqdm = None,
           distance: dst.DistanceMetric = None) -> Divik:
    """GAP limited DiviK with percentile initialization.

    Used in Master Thesis of Grzegorz Mrukwa.

    @param gap_trials: number of random datasets used in GAP statistic
    computation. Increases precision and computational overhead.
    @param distance_percentile: percentile of distance used for selection of
    initial representatives. Must be contained in range [0, 100] inclusive.
    Higher may reveal more nuances, but reduce robustness.
    @param iters_limit: limit of k-means iterations
    @param pool: pool for parallel processing. Recommended maxtasksperchild
    equal to number of cores.
    @param progress_reporter: tqdm-alike progress reporting object
    @param distance: distance metric
    @return: adjusted DiviK pipeline
    """
    assert 0 <= distance_percentile <= 100, distance_percentile
    if distance is None:
        distance = dst.SpearmanDistance()
    labeling = km.Labeling(distance)
    initialize = km.PercentileInitialization(distance, distance_percentile)
    kmeans = km._KMeans(labeling=km.Labeling(distance),
                        initialize=initialize,
                        number_of_iterations=iters_limit)
    best_kmeans_with_dunn = _dunn_optimized_kmeans(distance, kmeans, pool)
    fast_kmeans = partial(km._KMeans(labeling=labeling,
                                     initialize=initialize,
                                     number_of_iterations=10),
                          number_of_clusters=2)
    stop_if_split_makes_no_sense = st.combine(
        partial(st.minimal_size, size=20),
        st.Gap(distance, fast_kmeans, gap_trials, pool=pool))
    divik = partial(dv.divik,
                    split=best_kmeans_with_dunn,
                    feature_selectors=[_AMPLITUDE_FILTER, _VARIANCE_FILTER],
                    stop_condition=stop_if_split_makes_no_sense,
                    progress_reporter=progress_reporter,
                    min_features_percentage=.05)
    return divik


@_scenario
def basic(gap_trials: int = 100,
          distance_percentile: float = 99.,
          iters_limit: int = 100,
          distance: str = None,
          minimal_size: int = 20,
          rejection_size: int = None,
          rejection_percentage: float = None,
          minimal_features_percentage: float = .01,
          fast_kmeans_iters: int = 10,
          k_max: int = 10,
          correction_of_gap: bool = True,
          normalize_rows: bool = False,
          use_logfilters: bool = False,
          pool: Pool = None,
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
    if rejection_percentage is None and rejection_size is None:
        rejection_size = 0
    distance = dst.ScipyDistance(known_metrics[distance])
    labeling = km.Labeling(distance)
    initialize = km.PercentileInitialization(distance, distance_percentile)
    kmeans = km._KMeans(labeling=km.Labeling(distance),
                        initialize=initialize,
                        number_of_iterations=iters_limit,
                        normalize_rows=normalize_rows)
    best_kmeans_with_dunn = _dunn_optimized_kmeans(distance, kmeans, pool, k_max)
    fast_kmeans = partial(km._KMeans(labeling=labeling,
                                     initialize=initialize,
                                     number_of_iterations=fast_kmeans_iters,
                                     normalize_rows=normalize_rows),
                          number_of_clusters=2)
    stop_if_split_makes_no_sense = st.Gap(distance=distance,
                                          split_into_two=fast_kmeans,
                                          n_trials=gap_trials,
                                          correction=correction_of_gap,
                                          pool=pool)
    rejections = [
        partial(rj.reject_if_clusters_smaller_than, size=rejection_size,
                percentage=rejection_percentage)
    ]
    if use_logfilters:
        filters = [_LOG_AMPLITUDE_FILTER, _LOG_VARIANCE_FILTER]
    else:
        filters = [_AMPLITUDE_FILTER, _VARIANCE_FILTER]
    divik = partial(dv.divik,
                    split=best_kmeans_with_dunn,
                    feature_selectors=filters,
                    stop_condition=stop_if_split_makes_no_sense,
                    rejection_conditions=rejections,
                    progress_reporter=progress_reporter,
                    min_features_percentage=minimal_features_percentage,
                    prefiltering_stop_condition=partial(
                        st.minimal_size, size=max(k_max, minimal_size)))
    return divik
