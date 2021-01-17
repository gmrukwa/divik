import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClusterMixin,
    clone,
)

from divik.core import Subsets, configurable

_DEFAULT = object()


def _get_first_attr(obj, prop_name_candidates, default=_DEFAULT):
    for prop_name in prop_name_candidates:
        try:
            return getattr(obj, prop_name)
        except AttributeError:
            pass  # purposeful silence
    if default is _DEFAULT:
        raise AttributeError(
            f"{prop_name_candidates} do not exist in {obj.__class__.__name__}"
        )
    return default


def _get_final_estimator(estimator):
    try:  # sklearn.Pipeline or similar
        return estimator[-1]
    except TypeError:
        return estimator


@configurable
class TwoStep(BaseEstimator, ClusterMixin):
    """Perform a two-step clustering with a given clusterer

    Separates a dataset into ``n_subsets``, processes each of them separately
    and then combines the results.
    
    Works with centroid-based clustering methods, as it requires cluster
    representatives to combine the result.

    Parameters
    ----------
    clusterer : Union[AutoKMeans, Pipeline, KMeans]
        A centroid-based estimator for the purpose of clustering.

    n_subsets : int, default 10
        The number of subsets into which the original dataset should be
        separated
    
    random_state : int, default 42
        Random state to use for seeding the random number generator.
    
    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from divik.cluster import KMeans, TwoStep
    >>> X, _ = make_blobs(
    ...     n_samples=10_000, n_features=2, centers=3, random_state=42
    ... )
    >>> kmeans = KMeans(n_clusters=3)
    >>> ctr = TwoStep(kmeans).fit(X)
    """
    def __init__(self, clusterer, n_subsets: int = 10, random_state: int = 42):
        self.clusterer = clusterer
        self.n_subsets = n_subsets
        self.random_state = random_state

    def _label_in_subsets(self, X):
        subsets = Subsets(n_splits=self.n_subsets, random_state=self.random_state)
        X_sct = subsets.scatter(X)

        labels_part = [clone(self.clusterer).fit_predict(X_) for X_ in X_sct]
        sct_groups = [l * 0 + i for i, l in enumerate(labels_part)]

        labels = subsets.combine(labels_part)
        groups = subsets.combine(sct_groups)

        cross_fold_labels = [f"g{g}_l{l}" for l, g in zip(labels, groups)]
        return cross_fold_labels

    def fit(self, X, y=None):
        initial_labels = self._label_in_subsets(X)
        centroids = pd.DataFrame(X).groupby(initial_labels).mean()
        self.estimator_ = clone(self.clusterer)
        centroids_labels = self.estimator_.fit_predict(centroids)
        to_final = dict(zip(centroids.index, centroids_labels))
        final_labels = np.array([to_final[l] for l in initial_labels])
        self.labels_ = final_labels
        self.n_clusters_ = _get_first_attr(
            _get_final_estimator(self.estimator_),
            ['n_clusters', 'n_clusters_'],
        )
        return self

    def predict(self, X, y=None):
        return self.estimator_.predict(X)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_
