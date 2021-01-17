"""Reusable utilities used for building divik library"""

from ._cache import cached_fit
from ._gin_compat import (
    configurable,
    dump_gin_args,
    parse_args,
)
from ._parallel import (
    get_n_jobs,
    maybe_pool,
    share,
)
from ._seed import seed, seeded
from ._subsets import Subsets
from ._types import (
    Centroids,
    Data,
    DivikResult,
    IntLabels,
    SegmentationMethod,
)
from ._utils import (
    build,
    context_if,
    normalize_rows,
    visualize,
)

__all__ = [
    "cached_fit",
    "Centroids",
    "Data",
    "DivikResult",
    "IntLabels",
    "SegmentationMethod",
    "build",
    "context_if",
    "normalize_rows",
    "visualize",
    "get_n_jobs",
    "maybe_pool",
    "share",
    "seed",
    "seeded",
    "Subsets",
    "configurable",
    "dump_gin_args",
    "parse_args",
]
