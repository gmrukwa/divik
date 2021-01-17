import joblib
from functools import wraps

from ._gin_compat import configurable


@configurable
def cache_path(path: str = "cache"):
    # This is required to make the cache path configurable
    return path


def _is_computed(fieldname):
    return (
        fieldname.endswith("_")
        and not fieldname.endswith("__")
        and not fieldname.startswith("_")
    )


def cached_fit(cls):
    """Decorate a sklearn-compatible estimator to cache the fitting result

    It is a wrapper over joblib.Memory.cache, that supports runtime cache
    path definition.

    Set path definition through gin config with ``cache_path.path``
    identifier.
    """
    _fit = cls.fit

    @wraps(_fit)
    def fit(self, X, y=None):
        # This is pushed inside for the sake of parameters parsing
        memory = joblib.Memory(location=cache_path())
        cached = memory.cache(_fit)
        output = cached(self, X, y)
        computed_fields = [f for f in dir(output) if _is_computed(f)]
        for field in computed_fields:
            setattr(self, field, getattr(output, field))
        return self

    cls.fit = fit
    return cls
