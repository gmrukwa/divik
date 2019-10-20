from typing import Callable, Tuple, Dict, NamedTuple, List, Optional

import numpy as np
from skimage.color import label2rgb

Table = np.ndarray  # 2D matrix
Data = Table
Centroids = Table
IntLabels = np.ndarray
BoolFilter = np.ndarray
Quality = float
SegmentationMethod = Callable[[Data], Tuple[IntLabels, Centroids]]
ScoredSegmentation = Tuple[IntLabels, Centroids, Quality]
SelfScoringSegmentation = Callable[[Data], ScoredSegmentation]
StopCondition = Callable[[Data], bool]
Filter = Callable[[Data], Tuple[BoolFilter, float]]
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


def normalize_rows(data: Data) -> Data:
    normalized = data - data.mean(axis=1)[:, np.newaxis]
    norms = np.sum(np.abs(normalized) ** 2, axis=-1, keepdims=True)**(1./2)
    normalized /= norms
    return normalized


def visualize(label, xy, shape=None):
    x, y = xy.T
    if shape is None:
        shape = np.max(y) + 1, np.max(x) + 1
    y = y.max() - y
    label = label - label.min() + 1
    label_map = np.zeros(shape, dtype=int)
    label_map[y, x] = label
    image = label2rgb(label_map, bg_label=0)
    return image
