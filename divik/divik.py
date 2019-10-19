"""DiviK algorithm implementation.

divik.py

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
import gc
import logging as lg
from typing import List, Optional

import numpy as np
import tqdm

import divik.feature_selection as fs
from divik.types import \
    Data, \
    SelfScoringSegmentation, \
    StopCondition, \
    DivikResult
import divik.rejection as rj


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

    def filter(self, subset):
        lg.info('Feature filtering.')
        if lg.getLogger().getEffectiveLevel() <= lg.DEBUG:
            lg.debug('Subset shape: {0}'.format(subset.shape))
            lg.debug('Has NaNs: {0}'.format(np.isnan(subset).any()))
            lg.debug('Limits: min={0}; max={1}'.format(subset.min(), subset.max()))
            lg.debug('Has constant rows: {0}'.format(_constant_rows(subset)))

    def filtered(self, data, thresholds):
        lg.debug('Shape after filtering: {0}'.format(data.shape))
        lg.debug('Thresholds for filtering: {0}'.format(thresholds))
        constant = _constant_rows(data)
        if any(constant):
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
                   split: SelfScoringSegmentation,
                   feature_selectors: List[fs.FilteringMethod],
                   stop_condition: StopCondition,
                   rejection_conditions: List[rj.RejectionCondition],
                   report: _Reporter,
                   prefiltering_stop_condition: StopCondition,
                   min_features_percentage: float = .05) -> Optional[DivikResult]:
    subset = data[selection]

    if prefiltering_stop_condition(subset):
        report.finished_for(subset.shape[0])
        return None

    report.filter(subset)
    filters, thresholds, filtered_data = fs.select_sequentially(
        feature_selectors, subset, min_features_percentage)
    report.filtered(filtered_data, thresholds)

    report.stop_check()
    if stop_condition(filtered_data):
        report.finished_for(subset.shape[0])
        return None

    report.processing(filtered_data)
    partition, centroids, quality = split(filtered_data)
    if any(reject((partition, centroids, quality)) for reject in rejection_conditions):
        report.rejected(subset.shape[0])
        return None

    report.recurring(centroids.shape[0])
    recurse = partial(_divik_backend, data=data, split=split,
                      feature_selectors=feature_selectors,
                      stop_condition=stop_condition,
                      rejection_conditions=rejection_conditions,
                      report=report,
                      min_features_percentage=min_features_percentage,
                      prefiltering_stop_condition=prefiltering_stop_condition)
    del subset
    del filtered_data
    gc.collect()
    subregions = [
        recurse(selection=_recursive_selection(selection, partition, cluster))
        for cluster in np.unique(partition)
    ]

    report.assemble()
    return DivikResult(
        centroids=centroids,
        quality=quality,
        partition=partition,
        filters=filters,
        thresholds=thresholds,
        merged=partition,
        subregions=subregions
    )


def divik(data: Data, split: SelfScoringSegmentation,
          feature_selectors: List[fs.FilteringMethod],
          stop_condition: StopCondition,
          min_features_percentage: float = .05,
          progress_reporter: tqdm.tqdm = None,
          rejection_conditions: List[rj.RejectionCondition] = None,
          prefiltering_stop_condition: StopCondition = None) \
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
    if rejection_conditions is None:
        rejection_conditions = []
    report = _Reporter(progress_reporter)
    select_all = np.ones(shape=(data.shape[0],), dtype=bool)
    if prefiltering_stop_condition is None:
        def prefiltering_stop_condition(data: Data) -> bool:
            return False
    return _divik_backend(data,
                          selection=select_all,
                          split=split,
                          feature_selectors=feature_selectors,
                          stop_condition=stop_condition,
                          rejection_conditions=rejection_conditions,
                          report=report,
                          min_features_percentage=min_features_percentage,
                          prefiltering_stop_condition=prefiltering_stop_condition)
