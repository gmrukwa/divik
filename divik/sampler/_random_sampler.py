import numpy as np
from sklearn.preprocessing import MinMaxScaler

from divik._seeding import seed as seed_
from divik.feature_extraction import KneePCA
from ._core import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(self, n_samples: int = None):
        self.n_samples = n_samples

    def fit(self, X, y=None):
        if self.n_samples is None:
            n_samples = X.shape[0]
        else:
            n_samples = self.n_samples
        self.shape = n_samples, X.shape[1]
        self.scaler_ = MinMaxScaler().fit(X)
        return self
    
    def get_sample(self, seed):
        with seed_(seed):
            unscaled = np.random.random_sample(self.shape)
        return self.scaler_.inverse_transform(unscaled)


class RandomPCASampler(BaseSampler):
    def __init__(self, n_samples: int = None, whiten: bool = False):
        self.n_samples = n_samples
        self.whiten = whiten

    def fit(self, X, y=None):
        self.pca_ = KneePCA(self.whiten)
        transformed = self.pca_.fit_transform(X)
        self.sampler_ = RandomSampler(self.n_samples).fit(transformed)
        return self

    def get_sample(self, seed):
        transformed = self.sampler_.get_sample(seed)
        return self.pca_.inverse_transform(transformed)
