"""DiviK algorithm implementation"""
from functools import partial
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
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
                         min_features_percentage: float=.05) \
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


def divik(data: Data, split: SelfScoringSegmentation,
          feature_selectors: List[FilteringMethod],
          stop_condition: StopCondition,
          min_features_percentage: float=.05,
          progress_reporter: tqdm=None) -> Optional[DivikResult]:
    """Deglomerative intelligent segmentation framework

    @param data: dataset to segment
    @param split: unsupervised method of segmentation into some clusters
    @param feature_selectors: list of methods for feature selection
    @param stop_condition: criterion stating whether it is reasonable to split
    @param min_features_percentage: minimal percentage of preserved features
    @param progress_reporter: optional tqdm instance to report progress
    @return: result of segmentation if not stopped
    """
    filters, thresholds, filtered_data = _select_sequentially(
        feature_selectors, data, min_features_percentage)
    if stop_condition(filtered_data):
        if progress_reporter is not None:
            progress_reporter.update(data.shape[0])
        return None
    partition, centroids, quality = split(filtered_data)
    recurse = partial(divik, split=split,
                      feature_selectors=feature_selectors,
                      stop_condition=stop_condition)
    subregions = [
        recurse(cluster.values)
        for _, cluster in pd.DataFrame(data).groupby(partition)
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
