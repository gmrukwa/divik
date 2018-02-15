"""Numpy-based implementation of k-means algorithm"""
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

import spdivik.distance as dist


Labels = np.ndarray
Data = np.ndarray
Centroids = np.ndarray


class Initialization(object, metaclass=ABCMeta):
    """Initializes k-means algorithm"""
    @abstractmethod
    def __call__(self, data: Data, number_of_centroids: int) -> Centroids:
        """Generate initial centroids for k-means algorithm

        @param data: 2D matrix with observations in rows, features in columns
        @param number_of_centroids: number of centroids to be generated
        @return: centroids, in rows
        """
        raise NotImplementedError(self.__class__.__name__
                                  + " must implement __call__.")


class ExtremeInitialization(Initialization):
    """Initializes k-means by picking extreme points"""
    def __init__(self, distance: dist.DistanceMetric):
        self.distance = distance

    def __call__(self, data: Data, number_of_centroids: int) -> Centroids:
        """Generate initial centroids for k-means algorithm

        @param data: 2D matrix with observations in rows, features in columns
        @param number_of_centroids: number of centroids to be generated
        @return: centroids, in rows
        """
        if number_of_centroids > data.shape[0]:
            raise ValueError("Number of centroids (%i) greater than number of "
                             "observations (%i)."
                             % (number_of_centroids, data.shape[0]))
        features = data.T
        assumed_ys = features[0]
        modelled_xs = np.hstack([np.ones((data.shape[0], 1)),
                                 features[1:].T])
        default_singular_value_threshold = -1
        coefficients, _, _, _ = np.linalg.lstsq(
            modelled_xs, assumed_ys, rcond=default_singular_value_threshold)
        residuals = np.abs(np.dot(modelled_xs, coefficients) - assumed_ys)

        centroids = np.nan * np.zeros((number_of_centroids, data.shape[1]))
        centroids[0] = data[np.argmax(residuals)]

        for i in range(1, number_of_centroids):
            distances = self.distance(data, centroids[0:i])
            distances = np.min(distances, axis=1)
            centroids[i] = data[np.argmax(distances)]

        return centroids


class Labeling(object):
    """Labels observations by closest centroids"""
    def __init__(self, distance_metric: dist.DistanceMetric):
        """
        @param distance_metric: distance metric for estimation of closest
        """
        self.distance_metric = distance_metric

    def __call__(self, data: Data, centroids: Centroids) -> Labels:
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


def redefine_centroids(data: Data, labeling: Labels) -> Centroids:
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


class KMeans(object):
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
            -> Tuple[Labels, Centroids]:
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise ValueError("data is expected to be 2D np.array")
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
