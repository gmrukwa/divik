from typing import Union

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from divik._seeding import seed as seed_
from ._core import BaseSampler


class StratifiedSampler(BaseSampler):
    def __init__(self, sample_size: Union[int, float] = 10000):
        self.sample_size = sample_size
    
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self
    
    def get_sample(self, seed):
        split = StratifiedShuffleSplit(
            n_splits=1, train_size=self.sample_size, random_state=seed)
        for idx, _ in split.split(self.X_, self.y_):
            return self.X_[idx]
