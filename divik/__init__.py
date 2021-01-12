try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

from ._summary import plot, reject_split


__all__ = [
    "__version__",
    "plot", "reject_split",
]
