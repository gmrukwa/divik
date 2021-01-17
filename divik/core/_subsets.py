import numpy as np
from sklearn.model_selection import KFold


class Subsets:
    """Scatter dataset to disjoint random subsets and combine them back

    Parameters
    ----------
    n_splits : int, default 10
        Number of subsets that will be generated.

    random_state : int, default 42
        Random state to use for seeding the random number generator.

    Examples
    --------
    >>> from divik.core import Subsets
    >>> subsets = Subsets(n_splits=10, random_state=42)
    >>> X_list = subsets.scatter(X)
    >>> len(X_list)
    10
    >>> # do some computations on each subset
    >>> y = subsets.combine(y_list)
    """

    def __init__(self, n_splits: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def _fold_indices(self, X):
        kfold = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        return [idx for _, idx in kfold.split(X)]

    def scatter(self, X):
        self._n_rows = X.shape[0]
        self._idx = self._fold_indices(X)
        return [X[idx] for idx in self._idx]

    def _validate_shape(self, X_list):
        shapes = [X.shape for X in X_list]
        if np.sum([s[0] for s in shapes]) != self._n_rows:
            raise ValueError("Rows number mismatch")

        dims = [len(s) for s in shapes]
        if np.min(dims) != np.max(dims):
            raise ValueError("Inconsistent ndim")

        data_dims = [s[1:] for s in shapes]
        if np.any(np.min(data_dims, axis=0) != np.max(data_dims, axis=0)):
            raise ValueError("Data shape mismatch")

    def combine(self, X_list):
        self._validate_shape(X_list)
        X = np.zeros((self._n_rows, *X_list[0].shape[1:]), dtype=X_list[0].dtype)
        for idx, X_part in zip(self._idx, X_list):
            X[idx] = X_part
        return X
