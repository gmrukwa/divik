"""Numpy-based implementation of k-means algorithm"""

from ._initialization import \
    ExtremeInitialization, \
    Initialization, \
    PercentileInitialization
from ._core import KMeans, Labeling
