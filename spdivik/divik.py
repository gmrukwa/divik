"""DiviK algorithm implementation"""
from functools import partial
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from spdivik.types import \
    Centroids, \
    IntLabels, \
    BoolFilter, \
    Data, \
    SegmentationMethod, \
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

    def __call__(self, data: Data):
        return self.strategy(data)


def _make_filters_and_thresholds(feature_selectors: List[FilteringMethod],
                                 data: Data) -> Tuple[Filters, Thresholds]:
    """Estimate filters and thresholds with strategies for given data"""
    filters, thresholds = {}, {}
    for selector in feature_selectors:
        key = selector.name
        filters[key], thresholds[key] = selector(data)
    return filters, thresholds


def _select_features(filters: Filters, data: Data) -> Data:
    """Select estimated features for given data"""
    selection = np.ones((data.shape[1],), dtype=bool)
    for selector in filters.values():
        selection = np.logical_and(selection, selector)
    return data[:, selection]


def divik(data: Data, split: SegmentationMethod,
          feature_selectors: List[FilteringMethod],
          stop_condition: StopCondition) -> Optional[DivikResult]:
    """Deglomerative intelligent segmentation framework

    @param data: dataset to segment
    @param split: unsupervised method of segmentation into some clusters
    @param feature_selectors: list of methods for feature selection
    @param stop_condition: criterion stating whether it is reasonable to split
    @return: result of segmentation if not stopped
    """
    if stop_condition(data):
        return None
    filters, thresholds = _make_filters_and_thresholds(feature_selectors, data)
    filtered_data = _select_features(filters, data)
    partition, centroids = split(filtered_data)
    recurse = partial(divik, split=split,
                      feature_selectors=feature_selectors,
                      stop_condition=stop_condition)
    subregions = [
        recurse(data[partition == cluster])
        for cluster in np.unique(partition)
    ]
    return DivikResult(
        centroids=centroids,
        quality=float('nan'),
        partition=partition,
        filters=filters,
        thresholds=thresholds,
        merged=partition,
        subregions=subregions
    )
