from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import uuid

from sklearn.base import BaseEstimator, clone


_DATA = {}


class BaseSampler(BaseEstimator, metaclass=ABCMeta):
    """Base class for all the samplers

    Sampler is Pool-safe, i.e. can simply store a dataset.
    It will not be serialized by pickle when going to another process,
    if handled properly.

    Before you spawn a pool, a data must be moved to a module-level
    variable. To simplify that process a contract has been prepared.
    You open a context and operate within a context:

    with sampler.parallel() as sampler_, Pool() as pool:
        pool.map(sampler_.get_sample, range(10))
    
    Keep in mind, that __iter__ and fit are not accessible in parallel
    context. __iter__ would yield the same values independently in
    all the workers. Now it needs to be done consciously and in
    well-though manner. fit could lead to a non-predictable behaviour.
    If you need the original sampler, you can get a clone (not fit to
    the data).
    """
    def __iter__(self):
        """Inifinitely return samples"""
        i = 0
        while True:
            yield self.get_sample(i)
            i += 1

    @abstractmethod
    def get_sample(self, seed):
        """Return specific sample
        
        sampler.get_sample(x) == sampler.get_sample(x)
        x != y should yield sampler.get_sample(x) != sampler.get_sample(y)
        """
        raise NotImplementedError("get_sample is not implemented")

    def fit(self, X, y=None):
        """Fit sampler to data
        
        It's a base for both supervised and unsupervised samplers.
        """
        return self

    @contextmanager
    def parallel(self):
        """Create parallel context for the sampler to operate"""
        global _DATA
        sampler = str(uuid.uuid4())
        _DATA[sampler] = self
        try:
            yield ParallelSampler(sampler)
        finally:
            del _DATA[sampler]


class ParallelSampler:
    """Helper class for sharing the sampler functionality"""
    def __init__(self, sampler: str):
        self.sampler = sampler
    
    def get_sample(self, seed):
        """Return specific sample"""
        return _DATA[self.sampler].get_sample(seed)
    
    def clone(self):
        """Clones the original sampler"""
        return clone(_DATA[self.sampler])
