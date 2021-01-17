from typing import (
    Callable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np

Table = np.ndarray  # 2D matrix
Data = Table
Centroids = Table
IntLabels = np.ndarray
SegmentationMethod = Callable[[Data], Tuple[IntLabels, Centroids]]
Clustering = Union["divik.cluster.GAPSearch", "divik.cluster.DunnSearch"]


class DivikResult(NamedTuple):
    """Result of DiviK clustering"""

    clustering: Clustering
    """Fitted automated clustering estimator"""
    feature_selector: "divik.feature_selection.StatSelectorMixin"
    """Fitted feature selector"""
    merged: IntLabels
    """Recursively merged clustering labels"""
    subregions: List[Optional["DivikResult"]]
    """DivikResults for all obtained subregions"""
