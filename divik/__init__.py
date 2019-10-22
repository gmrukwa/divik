__version__ = '2.0.0'

from ._sklearn import DiviK
from .kmeans import AutoKMeans, KMeans
from .feature_selection import GMMSelector, HighAbundanceAndVarianceSelector
