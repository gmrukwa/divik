"""Unsupervised feature extraction methods"""

from ._histogram import HistogramEqualization
from ._pca import KneePCA
from ._spectral import LocallyAdjustedRbfSpectralEmbedding

__all__ = [
    "HistogramEqualization",
    "KneePCA",
    "LocallyAdjustedRbfSpectralEmbedding",
]
