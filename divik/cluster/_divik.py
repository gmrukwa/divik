"""DiviK algorithm implementation.

_divik.py

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
from sklearn.base import clone
import tqdm

from divik._utils import Data, DivikResult


def _recursive_selection(current_selection: np.ndarray, partition: np.ndarray,
                         cluster_number: int) -> np.ndarray:
    selection = np.zeros(shape=current_selection.shape, dtype=bool)
    selection[current_selection] = partition == cluster_number
    return selection


def _constant_rows(matrix: np.ndarray) -> List[int]:
    is_constant = matrix.min(axis=1) == matrix.max(axis=1)
    return np.where(is_constant)[0]


class DivikReporter:
    def __init__(self, progress_reporter: tqdm.tqdm = None,
                 warn_const: bool = True):
        self.progress_reporter = progress_reporter
        self.paths_open = 1
        self.warn_const = warn_const

    def filter(self, subset):
        lg.info('Feature filtering.')
        if lg.getLogger().getEffectiveLevel() <= lg.DEBUG:
            lg.debug('Subset shape: {0}'.format(subset.shape))
            lg.debug('Has NaNs: {0}'.format(np.isnan(subset).any()))
            lg.debug('Limits: min={0}; max={1}'.format(
                subset.min(), subset.max()))
            lg.debug('Has constant rows: {0}'.format(_constant_rows(subset)))

    def filtered(self, data):
        lg.debug('Shape after filtering: {0}'.format(data.shape))
        constant = _constant_rows(data)
        if self.warn_const and any(constant):
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


AutoKMeans = 'divik.cluster.AutoKMeans'
StatSelector = 'divik.feature_selection.StatSelectorMixin'


# @gmrukwa: I could not find more readable solution than recursion for now.
def divik(data: Data, selection: np.ndarray,
          fast_kmeans: AutoKMeans, full_kmeans: AutoKMeans,
          feature_selector: StatSelector,
          minimal_size: int, rejection_size: int, report: DivikReporter) \
        -> Optional[DivikResult]:
    subset = data[selection]

    if subset.shape[0] <= max(full_kmeans.max_clusters, minimal_size):
        report.finished_for(subset.shape[0])
        return None

    report.filter(subset)
    feature_selector = clone(feature_selector)
    filtered_data = feature_selector.fit_transform(subset)
    report.filtered(filtered_data)

    report.stop_check()
    fast_kmeans = clone(fast_kmeans).fit(filtered_data)
    if fast_kmeans.fitted_ and fast_kmeans.n_clusters_ == 1:
        report.finished_for(subset.shape[0])
        return None

    report.processing(filtered_data)
    clusterer = clone(full_kmeans).fit(filtered_data)
    partition = clusterer.labels_
    _, counts = np.unique(partition, return_counts=True)

    if any(counts <= rejection_size):
        report.rejected(subset.shape[0])
        return None

    report.recurring(len(counts))
    recurse = partial(
        divik, data=data, fast_kmeans=fast_kmeans,
        full_kmeans=full_kmeans, feature_selector=feature_selector,
        minimal_size=minimal_size, rejection_size=rejection_size,
        report=report)
    del subset
    del filtered_data
    gc.collect()
    subregions = [
        recurse(selection=_recursive_selection(selection, partition, cluster))
        for cluster in np.unique(partition)
    ]

    report.assemble()
    return DivikResult(clustering=clusterer, feature_selector=feature_selector,
                       merged=partition, subregions=subregions)
