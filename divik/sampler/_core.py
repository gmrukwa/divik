from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from itertools import count

from sklearn.base import BaseEstimator, clone


class BaseSampler(BaseEstimator, metaclass=ABCMeta):
    """Base class for all the samplers

    Sampler is Pool-safe, i.e. can simply store a dataset.
    It will not be serialized by pickle when going to another process,
    if handled properly.

    Before you spawn a pool, a data must be moved to a module-level
    variable. To simplify that process a contract has been prepared.
    You open a context and operate within a context:

    >>> with sampler.parallel() as sampler_,
    ...         Pool(initializer=sampler_.initializer,
    ...              initargs=sampler_.initargs) as pool:
    ...     pool.map(sampler_.get_sample, range(10))

    Keep in mind, that __iter__ and fit are not accessible in parallel
    context. __iter__ would yield the same values independently in
    all the workers. Now it needs to be done consciously and in
    well-though manner. fit could lead to a non-predictable behaviour.
    If you need the original sampler, you can get a clone (not fit to
    the data).
    """

    def __iter__(self):
        """Iter through `n_samples` samples or infinitely if unspecified"""
        if hasattr(self, "n_samples") and self.n_samples is not None:
            samples = range(self.n_samples)
        else:
            samples = count()
        for i in samples:
            yield self.get_sample(i)

    @abstractmethod
    def get_sample(self, seed):
        """Return specific sample

        Following assumptions should be met:
        a) sampler.get_sample(x) == sampler.get_sample(x)
        b) x != y should yield sampler.get_sample(x) != sampler.get_sample(y)

        Parameters
        ----------
        seed : int
            The seed to use to draw the sample

        Returns
        -------
        sample : array_like, (*self.shape_)
            Returns the drawn sample
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
        yield ParallelSampler(self)


class ParallelSampler:
    """Helper class for sharing the sampler functionality"""

    def __init__(self, sampler: BaseSampler):
        self.sampler = sampler

    def get_sample(self, seed):
        """Return specific sample"""
        return self.sampler.get_sample(seed)

    def initializer(self, *args):
        pass

    @property
    def initargs(self):
        return ()

    def clone(self):
        """Clones the original sampler"""
        return clone(self.sampler)
