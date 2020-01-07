from kneed import KneeLocator
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


# Implementation of PCA with data-driven variance explanation limit
def knee(explained_variance) -> int:
  limit = KneeLocator(x=np.arange(explained_variance.size, dtype=int),
                      y=explained_variance,
                      S=1.,
                      direction='increasing',
                      curve='concave')
  assert limit.knee is not None
  return limit.knee


class KneePCA(BaseEstimator, TransformerMixin):
    # TODO: Tests
    """Principal component analysis (PCA) with knee method
    PCA with automated components selection based on knee method
    over cumulative explained variance.

    Attributes
    ----------
    pca_ : PCA
        Fit PCA estimator.

    n_components_ : int
        The number of selected components.
    """
    def __init__(self, whiten: bool = False):
        self.whiten = whiten

    def fit(self, X, y=None):
    # TODO: Docs
        self.pca_ = PCA(n_components=None, copy=True, whiten=self.whiten,
                        svd_solver='full', tol=0.0, iterated_power='auto',
                        random_state=None).fit(X)
        self.n_components_ = knee(
            np.cumsum(self.pca_.explained_variance_ratio_))
        return self

    def transform(self, X, y=None):
    # TODO: Docs
        loads = self.pca_.transform(X)
        return loads[:, :self.n_components_]
    
    def inverse_transform(self, X):
    # TODO: Docs
        n_missing = self.pca_.n_components_ - self.n_components_
        missing = np.zeros(X.shape[0], n_missing)
        full = np.hstack([X, missing])
        return self.pca_.inverse_transform(full)
