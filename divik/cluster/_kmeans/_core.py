from typing import Tuple

import numpy as np
import scipy.spatial.distance as dst
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from divik.cluster._kmeans._initialization import \
    Initialization, \
    ExtremeInitialization, \
    PercentileInitialization
from divik._utils import (
    normalize_rows,
    Centroids,
    IntLabels,
    Data,
    SegmentationMethod,
)


class Labeling(object):
    """Labels observations by closest centroids"""
    def __init__(self, distance_metric: str):
        """
        @param distance_metric: distance metric for estimation of closest
        """
        self.distance_metric = distance_metric

    def __call__(self, data: Data, centroids: Centroids) -> IntLabels:
        """Find closest centroids

        @param data: observations in rows
        @param centroids: centroids in rows
        @return: vector of labels of centroids closest to points
        """
        if data.shape[1] != centroids.shape[1]:
            raise ValueError("Dimensionality of data and centroids must be "
                             "equal. Was %i and %i"
                             % (data.shape[1], centroids.shape[1]))
        distances = dst.cdist(data, centroids, self.distance_metric)
        return np.argmin(distances, axis=1)


def redefine_centroids(data: Data, labeling: IntLabels) -> Centroids:
    """Recompute centroids in data for given labeling

    @param data: observations
    @param labeling: partition of dataset into groups
    @return: centroids
    """
    if data.shape[0] != labeling.size:
        raise ValueError("Each observation must have label specified. Number "
                         "of labels: %i, number of observations: %i."
                         % (labeling.size, data.shape[0]))
    labels = np.unique(labeling)
    centroids = np.nan * np.zeros((len(labels), data.shape[1]))
    for label in labels:
        centroids[label] = np.mean(data[labeling == label], axis=0)
    return centroids


def _validate_kmeans_input(data: Data, number_of_clusters: int):
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        raise ValueError("data is expected to be 2D np.array")
    if number_of_clusters < 1:
        raise ValueError("number_of_clusters({0}) < 1".format(
            number_of_clusters))


def _validate_normalizable(data):
    is_constant = data.min(axis=1) == data.max(axis=1)
    if is_constant.any():
        constant_rows = np.where(is_constant)[0]
        msg = "Constant rows {0} are not allowed for normalization."
        raise ValueError(msg.format(constant_rows))


class _KMeans(SegmentationMethod):
    """K-means clustering"""
    def __init__(self, labeling: Labeling, initialize: Initialization,
                 number_of_iterations: int=100, normalize_rows: bool=False):
        """
        @param labeling: labeling method
        @param initialize: initialization method
        @param number_of_iterations: number of iterations
        @param normalize_rows: sets mean of row to 0 and norm to 1
        """
        self.labeling = labeling
        self.initialize = initialize
        self.number_of_iterations = number_of_iterations
        self.normalize_rows = normalize_rows

    def _fix_labels(self, data, centroids, labels, n_clusters):
        new_labels = labels.copy()
        known_labels = np.unique(labels)
        expected_labels = np.arange(n_clusters)
        missing_labels = np.setdiff1d(expected_labels, known_labels)
        new_centroids = np.nan * np.zeros((n_clusters, centroids.shape[1]))
        for known in known_labels:
            new_centroids[known] = centroids[known]
        for missing in missing_labels:
            new_center = np.nanmin(dst.cdist(
                data, new_centroids, metric=self.labeling.distance_metric
            ), axis=1).argmax()
            new_labels[new_center] = missing
            new_centroids[missing] = data[new_center]
        assert np.unique(new_labels).size == n_clusters
        return new_centroids, new_labels

    def __call__(self, data: Data, number_of_clusters: int) \
            -> Tuple[IntLabels, Centroids]:
        _validate_kmeans_input(data, number_of_clusters)
        if number_of_clusters == 1:
            return np.zeros((data.shape[0], 1), dtype=int), \
                   np.mean(data, axis=0, keepdims=True)
        data = data.reshape(data.shape, order='C')
        if self.normalize_rows:
            _validate_normalizable(data)
            data = normalize_rows(data)
        centroids = self.initialize(data, number_of_clusters)
        old_labels = np.nan * np.zeros((data.shape[0],))
        labels = self.labeling(data, centroids)
        for _ in range(self.number_of_iterations):
            if np.unique(labels).size != number_of_clusters:
                centroids, labels = self._fix_labels(
                    data, centroids, labels, number_of_clusters)
            if np.all(labels == old_labels):
                break
            old_labels = labels
            centroids = redefine_centroids(data, old_labels)
            labels = self.labeling(data, centroids)
        return labels, centroids


def _parse_initialization(name: str, distance: str,
                          percentile: float=None) -> Initialization:
    if name == 'percentile':
        return PercentileInitialization(distance, percentile)
    if name == 'extreme':
        return ExtremeInitialization(distance)
    raise ValueError('Unknown initialization: {0}'.format(name))


class KMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Means clustering

    Parameters
    ----------

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    distance : str, optional, default: 'euclidean'
        Distance measure. One of the distances supported by scipy package.

    init : {'percentile' or 'extreme'}
        Method for initialization, defaults to 'percentile':

        'percentile' : selects initial cluster centers for k-mean
        clustering starting from specified percentile of distance to
        already selected clusters

        'extreme': selects initial cluster centers for k-mean
        clustering starting from the furthest points to already specified
        clusters

    percentile : float, default: 95.0
        Specifies the starting percentile for 'percentile' initialization.
        Must be within range [0.0, 100.0]. At 100.0 it is equivalent to
        'extreme' initialization.

    max_iter : int, default: 100
        Maximum number of iterations of the k-means algorithm for a
        single run.

    normalize_rows : bool, default: False
        If True, rows are translated to mean of 0.0 and scaled to norm of 1.0.

    Attributes
    ----------

    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.

    labels_ :
        Labels of each point

    """
    # TODO: Add example of usage.
    def __init__(self, n_clusters: int, distance: str = 'euclidean',
                 init: str = 'percentile', percentile: float = 95.,
                 max_iter: int = 100, normalize_rows: bool = False):
        super().__init__()
        self.n_clusters = n_clusters
        self.distance = distance
        self.init = init
        self.percentile = percentile
        self.max_iter = max_iter
        self.normalize_rows = normalize_rows

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------

        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.

        y : Ignored
            not used, present here for API consistency by convention.
        """
        initialize = _parse_initialization(
            self.init, self.distance, self.percentile)
        kmeans = _KMeans(
            labeling=Labeling(self.distance),
            initialize=initialize,
            number_of_iterations=self.max_iter,
            normalize_rows=self.normalize_rows
        )
        self.labels_, self.cluster_centers_ = kmeans(
            X, number_of_clusters=self.n_clusters)
        self.labels_ = self.labels_.ravel()
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------

        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        if self.normalize_rows:
            X = normalize_rows(X)
        labels = dst.cdist(
            X, self.cluster_centers_, self.distance).argmin(axis=1)
        return labels

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------

        X_new : array, shape [n_samples, k]
            X transformed in the new space.

        """
        check_is_fitted(self)
        if self.normalize_rows:
            X = normalize_rows(X)
        return dst.cdist(X, self.cluster_centers_, self.distance)
