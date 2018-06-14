"""Reusable typings used across modules"""
from typing import Callable, Tuple

import numpy as np

Table = np.ndarray  # 2D matrix
Centroids = Table
IntLabels = np.ndarray
BoolFilter = np.ndarray
Data = Table
SegmentationMethod = Callable[[Data], Tuple[IntLabels, Centroids]]
StopCondition = Callable[[Data], bool]
Filter = Callable[[Data], Tuple[BoolFilter, float]]
