__version__ = '2.3.17'

from divik import core
from divik import feature_selection
from divik import feature_extraction
from divik import cluster
from divik import sampler
from ._summary import plot, reject_split


from .core import configurable
for __estimator in [
    feature_extraction.KneePCA,
    feature_extraction.LocallyAdjustedRbfSpectralEmbedding,
    feature_selection.GMMSelector,
    feature_selection.OutlierSelector,
    feature_selection.PercentageSelector,
    feature_selection.HighAbundanceAndVarianceSelector,
    feature_selection.NoSelector,
    feature_selection.OutlierAbundanceAndVarianceSelector,
    cluster.KMeans,
    cluster.GAPSearch,
    cluster.DunnSearch,
    cluster.DiviK,
    cluster.DunnDiviK,
    sampler.UniformSampler,
    sampler.UniformPCASampler,
    sampler.StratifiedSampler,
]:
    configurable(__estimator)


__all__ = [
    "__version__",
    "core",
    "cluster",
    "feature_selection",
    "feature_extraction",
    "sampler",
    "plot", "reject_split",
]
