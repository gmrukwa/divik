from typing import Union

from sklearn.model_selection import StratifiedShuffleSplit

from ._core import BaseSampler


class StratifiedSampler(BaseSampler):
    def __init__(self, n_rows: Union[int, float] = 10000,
                 n_samples: int = None):
        self.n_rows = n_rows
        self.n_samples = n_samples
    
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self
    
    def get_sample(self, seed):
        split = StratifiedShuffleSplit(
            n_splits=1, train_size=self.n_rows, random_state=seed)
        for idx, _ in split.split(self.X_, self.y_):
            return self.X_[idx]
