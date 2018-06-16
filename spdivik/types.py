"""Reusable typings used across modules"""
from typing import Callable, Tuple

import numpy as np

Table = np.ndarray  # 2D matrix
Centroids = Table
IntLabels = np.ndarray
BoolFilter = np.ndarray
Data = Table
Quality = float
SegmentationMethod = Callable[[Data], Tuple[IntLabels, Centroids]]
SelfScoringSegmentation = Callable[[Data], Tuple[IntLabels, Centroids, Quality]]
StopCondition = Callable[[Data], bool]
Filter = Callable[[Data], Tuple[BoolFilter, float]]
