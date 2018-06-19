from abc import ABCMeta, abstractmethod

import numpy as np

from spdivik import distance as dist
from spdivik.types import Data, Centroids


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
        if number_of_centroids < 1:
            raise ValueError('number_of_centroids({0}) < 1'.format(number_of_centroids))
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
    