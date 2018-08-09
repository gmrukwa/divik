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
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from spdivik.types import \
    Data, \
    SelfScoringSegmentation, \
    StopCondition, \
    Filter, \
    Filters, \
    Thresholds, \
    DivikResult
import spdivik.rejection as rj


class FilteringMethod:
    """Named filtering strategy."""

    def __init__(self, name: str, strategy: Filter):
        self.name = name
        self.strategy = strategy

    def __call__(self, data: Data, *args, **kwargs):
        """Filter data using embedded strategy."""
        return self.strategy(data, *args, **kwargs)


def _select_sequentially(feature_selectors: List[FilteringMethod], data: Data,
                         min_features_percentage: float = .05) \
        -> Tuple[Filters, Thresholds, Data]:
    filters, thresholds = {}, {}
    data_dimensionality = data.shape[1]
    minimal_dimensionality = int(min_features_percentage * data_dimensionality)
    current_selection = np.ones((data.shape[1],), dtype=bool)
    for selector in feature_selectors:
        key = selector.name
        filters[key], thresholds[key] = selector(
            data, min_features=minimal_dimensionality)
        data = data[:, filters[key]]
        current_selection[current_selection] = filters[key]
        filters[key] = current_selection
    return filters, thresholds, data


def _recursive_selection(current_selection: np.ndarray, partition: np.ndarray,
                         cluster_number: int) -> np.ndarray:
    selection = np.zeros(shape=current_selection.shape, dtype=bool)
    idx = 0
    for element_idx, element_selected in enumerate(current_selection):
        if element_selected:
            selection[element_idx] = partition[idx] == cluster_number
            idx += 1
    return selection


class _Reporter:
    def __init__(self, progress_reporter: tqdm = None):
        self.progress_reporter = progress_reporter
        self.paths_open = 1

    def filter(self):
        lg.info('Feature filtering.')

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
                   feature_selectors: List[FilteringMethod],
                   stop_condition: StopCondition,
                   rejection_conditions: List[rj.RejectionCondition],
                   report: _Reporter,
                   min_features_percentage: float = .05) -> Optional[DivikResult]:
    subset = data[selection]
    report.filter()
    filters, thresholds, filtered_data = _select_sequentially(
        feature_selectors, subset, min_features_percentage)
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
                      min_features_percentage=min_features_percentage)
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
          feature_selectors: List[FilteringMethod],
          stop_condition: StopCondition,
          min_features_percentage: float = .05,
          progress_reporter: tqdm = None,
          rejection_conditions: List[rj.RejectionCondition] = None) \
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
    @return: result of segmentation if not stopped
    """
    if rejection_conditions is None:
        rejection_conditions = []
    report = _Reporter(progress_reporter)
    return _divik_backend(data, np.ones(shape=(data.shape[0],), dtype=bool),
                          split=split, feature_selectors=feature_selectors,
                          stop_condition=stop_condition,
                          rejection_conditions=rejection_conditions,
                          report=report,
                          min_features_percentage=min_features_percentage)
