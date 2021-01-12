import uuid
from contextlib import contextmanager
from typing import Union

from sklearn.model_selection import StratifiedShuffleSplit

from divik.core import configurable, share

from ._core import BaseSampler, ParallelSampler

_DATA = {}


@configurable
class StratifiedSampler(BaseSampler):
    """Sample the original data preserving proportions of groups

    Parameters
    -----------
    n_rows : int or float, optional (default 10000)
        Allows to limit the number of rows in the drawn samples.
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the sample. If
        int, represents the absolute number of rows.

    n_samples : int, optional (default None)
        Allows to limit the number of samples when iterating

    Attributes
    ----------
    X_ : array_like, shape (n_rows, n_features)
        Data to sample from

    y_ : array_like, shape (n_rows,)
        Group labels
    """

    def __init__(self, n_rows: Union[int, float] = 100, n_samples: int = None):
        self.n_rows = n_rows
        self.n_samples = n_samples

    def fit(self, X, y):
        """Fit the model from data in X.

        Both inputs are preserved inside to sample from the data.

        Parameters
        ----------
        X : array-like, shape (n_rows, n_features)
            Training vector, where n_rows is the number of rows
            and n_features is the number of features.

        y: array-like, shape (n_rows,)

        Returns
        -------
        self : StratifiedSampler
            Returns the instance itself.
        """
        self.X_ = X
        self.y_ = y
        return self

    def get_sample(self, seed):
        """Return specific sample

        Sample is drawn from the set of existing rows. A proportion of
        gorups should be more-or-less the same, depending on the size
        of the sample.

        Parameters
        ----------
        seed : int
            The seed to use to draw the sample

        Returns
        -------
        sample : array_like, (*self.shape_)
            Returns the drawn sample
        """
        split = StratifiedShuffleSplit(
            n_splits=1, train_size=self.n_rows, random_state=seed
        )
        for idx, _ in split.split(self.X_, self.y_):
            return self.X_[idx]

    @contextmanager
    def parallel(self):
        global _DATA
        ref = str(uuid.uuid4())
        with share(self.X_) as X, share(self.y_) as y:
            _DATA[ref] = ((self.n_rows, self.n_samples), (X, y))
            try:
                yield StratifiedParallelSampler(ref)
            finally:
                del _DATA[ref]


class StratifiedParallelSampler(ParallelSampler):
    def __init__(self, ref):
        self._ref = ref

    @property
    def sampler(self):
        global _DATA
        init_args, fit_args = _DATA[self._ref]
        X, y = fit_args
        return StratifiedSampler(*init_args).fit(X.value, y.value)

    def get_sample(self, seed):
        return self.sampler.get_sample(seed)

    def initializer(self, *args):
        global _DATA
        _DATA[self._ref] = args

    @property
    def initargs(self):
        global _DATA
        return _DATA[self._ref]
