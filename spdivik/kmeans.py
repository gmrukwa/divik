"""Numpy-based implementation of k-means algorithm"""
from abc import ABCMeta, abstractmethod

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
