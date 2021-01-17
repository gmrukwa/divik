from functools import partial

import numpy as np
from skimage.exposure import cumulative_distribution
from sklearn.base import BaseEstimator, TransformerMixin

from divik.core import configurable, maybe_pool


@configurable
class HistogramEqualization(BaseEstimator, TransformerMixin):
    """Equalize histogram of the features to increase contrast
    
    Based on https://github.com/scikit-image/scikit-image/blob/master/skimage/exposure/exposure.py#L187-L223

    Parameters
    ----------

    n_bins : int, default 256
        Number of bins for histogram equalization.

    n_jobs : int, default -1
        Number of CPU cores to use during equalization

    Attributes
    ----------

    cdf_ : array
        Values of cumulative distribution function for all the features
    
    bins_ : array
        Bin centers for all the features
    """

    def __init__(self, n_bins: int = 256, n_jobs: int = -1):
        self.n_bins = n_bins
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        cdf = partial(cumulative_distribution, nbins=self.n_bins)
        with maybe_pool(processes=self.n_jobs) as pool:
            cdf_and_bins = pool.map(cdf, X.T)
        self.cdf_, self.bins_ = zip(*cdf_and_bins)

        return self

    def transform(self, X, y=None):
        features = zip(X.T, self.bins_, self.cdf_)
        with maybe_pool(processes=self.n_jobs) as pool:
            return np.transpose(pool.starmap(np.interp, features))
