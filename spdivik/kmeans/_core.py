from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from spdivik import distance as dist
from spdivik.kmeans._initialization import \
    Initialization, \
    ExtremeInitialization, \
    PercentileInitialization
from spdivik.types import Data, Centroids, IntLabels, SegmentationMethod


class Labeling(object):
    """Labels observations by closest centroids"""
    def __init__(self, distance_metric: dist.DistanceMetric):
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
        distances = self.distance_metric(data, centroids)
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


def _normalize_rows(data: Data) -> Data:
    data -= data.mean(axis=1)[:, np.newaxis]
    norms = np.sum(np.abs(data) ** 2, axis=-1, keepdims=True)**(1./2)
    data /= norms
    return data


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

    def __call__(self, data: Data, number_of_clusters: int) \
            -> Tuple[IntLabels, Centroids]:
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise ValueError("data is expected to be 2D np.array")
        if number_of_clusters < 1:
            raise ValueError("number_of_clusters({0}) < 1".format(
                number_of_clusters))
        elif number_of_clusters == 1:
            return np.zeros((data.shape[0], 1), dtype=int), \
                   np.mean(data, axis=0)
        if self.normalize_rows:
            data = _normalize_rows(data)
        centroids = self.initialize(data, number_of_clusters)
        old_labels = np.nan * np.zeros((data.shape[0],))
        labels = self.labeling(data, centroids)
        for _ in range(self.number_of_iterations):
            if np.all(labels == old_labels):
                break
            old_labels = labels
            centroids = redefine_centroids(data, old_labels)
            labels = self.labeling(data, centroids)
        return labels, centroids


def parse_distance(name: str) -> dist.ScipyDistance:
    known_distances = {metric.value: metric for metric in dist.KnownMetric}
    assert name in known_distances, \
        'Unknown distance {0}. Known: {1}'.format(
            name, list(known_distances.keys()))
    distance = dist.ScipyDistance(known_distances[name])
    return distance


def _parse_initialization(name: str, distance: dist.ScipyDistance,
                          percentile: float=None) -> Initialization:
    if name == 'percentile':
        return PercentileInitialization(distance, percentile)
    if name == 'extreme':
        return ExtremeInitialization(distance)
    raise ValueError('Unknown initialization: {0}'.format(name))


class KMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    # TODO: Add documentation
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
        dist = parse_distance(self.distance)
        initialize = _parse_initialization(self.init, dist, self.percentile)
        kmeans = _KMeans(
            labeling=Labeling(dist),
            initialize=initialize,
            number_of_iterations=self.max_iter,
            normalize_rows=self.normalize_rows
        )
        self.labels_, self.cluster_centers_ = kmeans(
            X, number_of_clusters=self.n_clusters)
        self.labels_ = self.labels_.ravel()
        return self

    def predict(self, X):
        check_is_fitted(self, 'cluster_centers_')
        distance = parse_distance(self.distance)
        labels = distance(X, self.cluster_centers_).argmin(axis=1)
        return labels

    def transform(self, X):
        check_is_fitted(self, 'cluster_centers_')
        distance = parse_distance(self.distance)
        distances = distance(X, self.cluster_centers_).min(axis=1)
        return distances
