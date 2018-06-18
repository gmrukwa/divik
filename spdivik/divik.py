"""DiviK algorithm implementation"""
from functools import partial
import gc
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from tqdm import tqdm

from spdivik.types import \
    Centroids, \
    IntLabels, \
    BoolFilter, \
    Data, \
    SelfScoringSegmentation, \
    StopCondition, \
    Filter

FilterName = str
Filters = Dict[FilterName, BoolFilter]
Thresholds = Dict[FilterName, float]
DivikResult = NamedTuple('DivikResult', [
    ('centroids', Centroids),
    ('quality', float),
    ('partition', IntLabels),
    ('filters', Filters),
    ('thresholds', Thresholds),
    ('merged', IntLabels),
    ('subregions', List[Optional['DivikResult']]),
])


class FilteringMethod:
    """Named filtering strategy"""
    def __init__(self, name: str, strategy: Filter):
        self.name = name
        self.strategy = strategy

    def __call__(self, data: Data, *args, **kwargs):
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


# TODO: ELIMINATE RECURSION ASAP!!!
def _divik_backend(data: Data, selection: np.ndarray,
                   split: SelfScoringSegmentation,
                   feature_selectors: List[FilteringMethod],
                   stop_condition: StopCondition,
                   min_features_percentage: float = .05,
                   progress_reporter: tqdm = None) -> Optional[DivikResult]:
    subset = data[selection]
    filters, thresholds, filtered_data = _select_sequentially(
        feature_selectors, subset, min_features_percentage)
    if stop_condition(filtered_data):
        if progress_reporter is not None:
            progress_reporter.update(subset.shape[0])
        return None
    partition, centroids, quality = split(filtered_data)
    recurse = partial(_divik_backend, split=split,
                      feature_selectors=feature_selectors,
                      stop_condition=stop_condition)
    del subset
    del filtered_data
    gc.collect()
    subregions = [
        recurse(data, _recursive_selection(selection, partition, cluster))
        for cluster in np.unique(partition)
    ]
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
          progress_reporter: tqdm = None) -> Optional[DivikResult]:
    """Deglomerative intelligent segmentation framework

    @param data: dataset to segment
    @param split: unsupervised method of segmentation into some clusters
    @param feature_selectors: list of methods for feature selection
    @param stop_condition: criterion stating whether it is reasonable to split
    @param min_features_percentage: minimal percentage of preserved features
    @param progress_reporter: optional tqdm instance to report progress
    @return: result of segmentation if not stopped
    """
    return _divik_backend(data, np.ones(shape=(data.shape[0],), dtype=bool),
                          split=split, feature_selectors=feature_selectors,
                          stop_condition=stop_condition,
                          min_features_percentage=min_features_percentage,
                          progress_reporter=progress_reporter)
