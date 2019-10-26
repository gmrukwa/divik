from abc import ABCMeta, abstractmethod

import numpy as np

from divik import _distance as dist
from divik._utils import Centroids, Data


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


def _find_residuals(data: Data) -> np.ndarray:
    features = data.T
    assumed_ys = features[0]
    modelled_xs = np.hstack([np.ones((data.shape[0], 1)),
                             features[1:].T])
    default_singular_value_threshold = -1
    coefficients, _, _, _ = np.linalg.lstsq(
        modelled_xs, assumed_ys, rcond=default_singular_value_threshold)
    residuals = np.abs(np.dot(modelled_xs, coefficients) - assumed_ys)
    return residuals


def _validate(data: Data, number_of_centroids: int):
    if number_of_centroids > data.shape[0]:
        raise ValueError("Number of centroids (%i) greater than number of "
                         "observations (%i)."
                         % (number_of_centroids, data.shape[0]))
    if number_of_centroids < 1:
        raise ValueError(
            'number_of_centroids({0}) < 1'.format(number_of_centroids))


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
        _validate(data, number_of_centroids)
        residuals = _find_residuals(data)
        centroids = np.nan * np.zeros((number_of_centroids, data.shape[1]))
        centroids[0] = data[np.argmax(residuals)]

        distances = np.inf * np.ones((data.shape[0], ))
        for i in range(1, number_of_centroids):
            current_distance = self.distance(data, centroids[np.newaxis, i - 1])
            distances[:] = np.minimum(current_distance.ravel(), distances)
            centroids[i] = data[np.argmax(distances)]

        return centroids


class PercentileInitialization(Initialization):
    def __init__(self, distance: dist.DistanceMetric, percentile: float=99.):
        assert 0 <= percentile <= 100, percentile
        self.distance = distance
        self.percentile = percentile

    def _get_percentile_element(self, values: np.ndarray) -> int:
        value = np.percentile(values, q=self.percentile,
                              interpolation='nearest')
        assert values.size > 0
        assert not np.isnan(values).any()
        matches = values == value
        assert np.any(matches), (value, values)
        return int(np.flatnonzero(matches)[0])

    def __call__(self, data: Data, number_of_centroids: int) -> Centroids:
        _validate(data, number_of_centroids)
        residuals = _find_residuals(data)
        selected = self._get_percentile_element(residuals)
        centroids = np.nan * np.zeros((number_of_centroids, data.shape[1]))
        centroids[0] = data[selected]
        assert not np.any(np.isnan(centroids[0]))

        distances = np.inf * np.ones((data.shape[0],))
        for i in range(1, number_of_centroids):
            assert not np.any(np.isnan(centroids[np.newaxis, i - 1]))
            current_distance = self.distance(data, centroids[np.newaxis, i - 1])
            nans = np.isnan(current_distance)
            if np.any(nans):
                locations_of_nans = np.array(list(zip(*np.nonzero(nans))))
                raise ValueError('Distances between points cannot be NaN. '
                                 + 'This indicates that your data is probably'
                                 + ' corrupted and analysis cannot be continued'
                                 + ' in this setting. '
                                 + 'Amount of NaNs: {0}. '.format(nans.sum())
                                 + 'At positions described by [spot, centroid]: '
                                 + '{0}'.format(locations_of_nans))
            distances[:] = np.minimum(current_distance.ravel(), distances)
            selected = self._get_percentile_element(distances)
            centroids[i] = data[selected]

        return centroids
