__version__ = '2.3.14'

from ._seeding import seeded
from ._utils import DivikResult
from divik import feature_selection
from divik import feature_extraction
from divik import cluster
from divik import sampler
from ._summary import plot, reject_split
from ._gin_compat import (
    configurable,
    parse_gin_args
)

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
    "cluster",
    "feature_selection",
    "feature_extraction",
    "sampler",
    "seeded",
    "configurable", "parse_gin_args",
    'DivikResult',
    "plot", "reject_split",
]
