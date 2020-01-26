from typing import (
    Callable,
    Tuple,
    NamedTuple,
    List,
    Optional,
    Union,
)

import numpy as np

Table = np.ndarray  # 2D matrix
Data = Table
Centroids = Table
IntLabels = np.ndarray
SegmentationMethod = Callable[[Data], Tuple[IntLabels, Centroids]]
Clustering = Union['divik.cluster.GAPSearch', 'divik.cluster.DunnSearch']
DivikResult = NamedTuple('DivikResult', [
    ('clustering', Clustering),
    ('feature_selector', 'divik.feature_selection.StatSelectorMixin'),
    ('merged', IntLabels),
    ('subregions', List[Optional['DivikResult']]),
])
