import os
from typing import Callable, Tuple, NamedTuple, List, Optional

import numpy as np
from skimage.color import label2rgb

Table = np.ndarray  # 2D matrix
Data = Table
Centroids = Table
IntLabels = np.ndarray
Quality = float
SegmentationMethod = Callable[[Data], Tuple[IntLabels, Centroids]]
DivikResult = NamedTuple('DivikResult', [
    ('clustering', 'divik.AutoKMeans'),
    ('feature_selector', 'divik.feature_selection.HighAbundanceAndVarianceSelector'),
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


def get_n_jobs(n_jobs):
    n_cpu = os.cpu_count() or 1
    n_jobs = 1 if n_jobs is None else n_jobs
    if n_jobs <= 0:
        n_jobs = min(n_jobs + 1 + n_cpu, n_cpu)
    n_jobs = n_jobs or n_cpu
    return n_jobs
