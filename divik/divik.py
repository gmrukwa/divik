"""DiviK algorithm implementation.

divik.py

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
import gc
import logging as lg
from typing import List, Optional

import numpy as np
import tqdm

import divik.feature_selection as fs
import divik.kmeans as km
from divik.utils import Data, DivikResult


def _recursive_selection(current_selection: np.ndarray, partition: np.ndarray,
                         cluster_number: int) -> np.ndarray:
    selection = np.zeros(shape=current_selection.shape, dtype=bool)
    selection[current_selection] = partition == cluster_number
    return selection


def _constant_rows(matrix: np.ndarray) -> List[int]:
    is_constant = matrix.min(axis=1) == matrix.max(axis=1)
    return np.where(is_constant)[0]


class _Reporter:
    def __init__(self, progress_reporter: tqdm.tqdm = None):
        self.progress_reporter = progress_reporter
        self.paths_open = 1
        self.warn_const = True

    def filter(self, subset):
        lg.info('Feature filtering.')
        if lg.getLogger().getEffectiveLevel() <= lg.DEBUG:
            lg.debug('Subset shape: {0}'.format(subset.shape))
            lg.debug('Has NaNs: {0}'.format(np.isnan(subset).any()))
            lg.debug('Limits: min={0}; max={1}'.format(subset.min(), subset.max()))
            lg.debug('Has constant rows: {0}'.format(_constant_rows(subset)))

    def filtered(self, data):
        lg.debug('Shape after filtering: {0}'.format(data.shape))
        constant = _constant_rows(data)
        if any(constant) and self.warn_const:
            msg = 'After feature filtering some rows are constant: {0}. ' \
                  'This may not work with specific configurations.'
            lg.warning(msg.format(constant))

    def stop_check(self):
        lg.info('Stop condition check.')

    def finished_for(self, n_observations: int):
        self.paths_open -= 1
        lg.info('Stop condition fired for {0}. {1} paths open.'
                .format(n_observations, self.paths_open))
        if self.progress_reporter is not None:
            self.progress_reporter.update(n_observations)

    def rejected(self, n_observations: int):
        self.paths_open -= 1
        lg.info('Rejected segmentation of {0}. {1} paths open.'
                .format(n_observations, self.paths_open))
        if self.progress_reporter is not None:
            self.progress_reporter.update(n_observations)

    def processing(self, dataset: np.ndarray):
        lg.info('Processing subset with {0} observations and {1} features.'
                .format(*dataset.shape))

    def recurring(self, n_subregions):
        self.paths_open += n_subregions
        lg.info('Recurring into {0} subregions. {1} paths open.'
                .format(n_subregions, self.paths_open))

    def assemble(self):
        self.paths_open -= 1
        lg.info('Assembled. {0} paths open.'.format(self.paths_open))


# @gmrukwa: I could not find more readable solution than recursion for now.
def _divik_backend(data: Data, selection: np.ndarray,
                   fast_kmeans_iters: int,
                   k_max: int, n_jobs: int, distance: str,
                   distance_percentile: float, iters_limit: int,
                   normalize_rows: bool,
                   minimal_size: int,
                   rejection_size: int,
                   report: _Reporter,
                   random_seed: int = 0,
                   gap_trials: int = 10,
                   min_features_percentage: float = .05,
                   use_logfilters: bool = False) -> Optional[DivikResult]:
    subset = data[selection]

    if subset.shape[0] <= max(k_max, minimal_size):
        report.finished_for(subset.shape[0])
        return None

    report.filter(subset)
    feature_selector = fs.HighAbundanceAndVarianceSelector(
        use_log=use_logfilters, min_features_rate=min_features_percentage)
    filtered_data = feature_selector.fit_transform(subset)
    report.filtered(filtered_data)

    report.stop_check()
    fast_kmeans = km.AutoKMeans(
        max_clusters=2, n_jobs=n_jobs, method="gap", distance=distance,
        init='percentile', percentile=distance_percentile,
        max_iter=iters_limit, normalize_rows=normalize_rows,
        gap={"max_iter": fast_kmeans_iters, "seed": random_seed,
             "trials": gap_trials, "correction": True},
        verbose=False).fit(filtered_data)
    if fast_kmeans.fitted_ and fast_kmeans.n_clusters_ == 1:
        report.finished_for(subset.shape[0])
        return None

    report.processing(filtered_data)
    clusterer = km.AutoKMeans(
        max_clusters=k_max, min_clusters=2, n_jobs=n_jobs, method='dunn',
        distance=distance, init='percentile', percentile=distance_percentile,
        max_iter=iters_limit, normalize_rows=normalize_rows, gap=None,
        verbose=False
    ).fit(filtered_data)
    partition = clusterer.labels_
    _, counts = np.unique(partition, return_counts=True)

    if any(counts <= rejection_size):
        report.rejected(subset.shape[0])
        return None

    report.recurring(len(counts))
    recurse = partial(_divik_backend, data=data,
                      fast_kmeans_iters=fast_kmeans_iters,
                      k_max=k_max,
                      n_jobs=n_jobs,
                      distance=distance,
                      distance_percentile=distance_percentile,
                      iters_limit=iters_limit,
                      normalize_rows=normalize_rows,
                      minimal_size=minimal_size,
                      rejection_size=rejection_size,
                      report=report,
                      random_seed=random_seed,
                      gap_trials=gap_trials,
                      min_features_percentage=min_features_percentage,
                      use_logfilters=use_logfilters)
    del subset
    del filtered_data
    gc.collect()
    subregions = [
        recurse(selection=_recursive_selection(selection, partition, cluster))
        for cluster in np.unique(partition)
    ]

    report.assemble()
    return DivikResult(
        clustering=clusterer,
        feature_selector=feature_selector,
        merged=partition,
        subregions=subregions
    )


# TODO: consider how to push the pool without re-creating it.
def divik(data: Data,
          fast_kmeans_iters: int,
          k_max: int, n_jobs: int, distance: str,
          distance_percentile: float, iters_limit: int,
          normalize_rows: bool,
          random_seed: int,
          gap_trials: int,
          min_features_percentage: float = .05,
          progress_reporter: tqdm.tqdm = None,
          minimal_size: int = 2,
          rejection_size: int = 0,
          use_logfilters: bool = False) \
        -> Optional[DivikResult]:
    """Deglomerative intelligent segmentation framework.

    @param data: dataset to segment
    @param split: unsupervised method of segmentation into some clusters
    @param feature_selectors: list of methods for feature selection
    @param stop_condition: criterion stating whether it is reasonable to split
    @param min_features_percentage: minimal percentage of preserved features
    @param progress_reporter: optional tqdm instance to report progress
    @param rejection_conditions: optional list of conditions that reject
    clustering result
    @param prefiltering_stop_condition: stop condition that could be applied
    without feature filtering
    @return: result of segmentation if not stopped
    """
    if np.isnan(data).any():
        raise ValueError("NaN values are not supported.")
    report = _Reporter(progress_reporter)
    select_all = np.ones(shape=(data.shape[0],), dtype=bool)
    return _divik_backend(data,
                          selection=select_all,
                          fast_kmeans_iters=fast_kmeans_iters,
                          k_max=k_max,
                          n_jobs=n_jobs,
                          distance=distance,
                          distance_percentile=distance_percentile,
                          iters_limit=iters_limit,
                          normalize_rows=normalize_rows,
                          minimal_size=minimal_size,
                          rejection_size=rejection_size,
                          report=report,
                          random_seed=random_seed,
                          gap_trials=gap_trials,
                          min_features_percentage=min_features_percentage,
                          use_logfilters=use_logfilters)
