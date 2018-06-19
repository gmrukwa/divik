from typing import Tuple

import numpy as np

from spdivik import distance as dist
from spdivik.kmeans import Initialization
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


class KMeans(SegmentationMethod):
    """K-means clustering"""
    def __init__(self, labeling: Labeling, initialize: Initialization,
                 number_of_iterations: int=100):
        """
        @param labeling: labeling method
        @param initialize: initialization method
        @param number_of_iterations: number of iterations
        """
        self.labeling = labeling
        self.initialize = initialize
        self.number_of_iterations = number_of_iterations

    def __call__(self, data: Data, number_of_clusters: int) \
            -> Tuple[IntLabels, Centroids]:
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise ValueError("data is expected to be 2D np.array")
        if number_of_clusters < 1:
            raise ValueError("number_of_clusters({0}) < 1".format(number_of_clusters))
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
