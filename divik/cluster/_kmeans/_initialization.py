from abc import ABCMeta, abstractmethod
from typing import List, NamedTuple, Union

import numpy as np
import scipy.spatial.distance as dist
from skimage.filters import threshold_otsu
from sklearn.linear_model import LinearRegression

from divik.core import Centroids, Data


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


def _find_residuals(data: Data, sample_weight=None) -> np.ndarray:
    features = data.T
    assumed_ys = features[0]
    modelled_xs = np.hstack([np.ones((data.shape[0], 1)),
                            features[1:].T])
    lr = LinearRegression().fit(modelled_xs, assumed_ys,
                                sample_weight=sample_weight)
    residuals = np.abs(lr.predict(modelled_xs) - assumed_ys)
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
    def __init__(self, distance: str):
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
            current_distance = dist.cdist(
                data, centroids[np.newaxis, i - 1], self.distance)
            distances[:] = np.minimum(current_distance.ravel(), distances)
            centroids[i] = data[np.argmax(distances)]

        return centroids


class PercentileInitialization(Initialization):
    def __init__(self, distance: str, percentile: float=99.):
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
            current_distance = dist.cdist(
                data, centroids[np.newaxis, i - 1], self.distance)
            nans = np.isnan(current_distance)
            if np.any(nans):
                locations_of_nans = np.array(list(zip(*np.nonzero(nans))))
                raise ValueError('Distances between points cannot be NaN. '
                                 + 'This indicates that your data is probably'
                                 + ' corrupted and analysis cannot be '
                                 + 'continued in this setting. '
                                 + 'Amount of NaNs: {0}. '.format(nans.sum())
                                 + 'At positions described by [spot, '
                                 + 'centroid]: {0}'.format(locations_of_nans))
            distances[:] = np.minimum(current_distance.ravel(), distances)
            selected = self._get_percentile_element(distances)
            centroids[i] = data[selected]

        return centroids


class Leaf(NamedTuple):
    bounds: np.ndarray
    centroid: np.ndarray
    count: int = 0

KDTree = Union['Node', Leaf]
        
class Node(NamedTuple):
    pivot_value: float
    pivot_feature: int
    left: KDTree = None
    right: KDTree = None


def make_tree(X, leaf_size: Union[int, float]=0.01, pivot=threshold_otsu) -> KDTree:
    """Make KDTree out of the data

    Construct a KDTree out of data using Otsu threshold as a pivoting element.
    Each split makes two segments. The result doesn't contain the original
    data, just the splitting points, bounds of leaves, centroids in each box
    and count of items.

    Parameters
    ==========
    X : array_like, (n_samples, n_features)
        Set of observations to divide into boxes
        
    leaf_size : int or float, optional (default 0.01)
        Desired leaf size. When int, it will be between `leaf_size` and
        `2 * leaf_size`. When float, it will be between
        `leaf_size * n_samples` and `2 * leaf_size * n_samples`
    
    pivot : callable, optional (default skimage.threshold_otsu)
        Method to find the pivot element. Recommended are:
        - skimage.threshold_otsu
        - np.median
        - np.mean
    
    Returns
    =======
    tree : KDTree
        Lightweight KD-Tree over the data
    """
    X = np.asanyarray(X)
    if isinstance(leaf_size, float):
        if 0 <= leaf_size <= 1:
            leaf_size = max(int(leaf_size * X.shape[0]), 1)
        else:
            raise ValueError('leaf_size must be between 0 and 1 when float')
    if X.shape[0] < 2 * leaf_size:
        bounds = np.vstack([X.min(axis=0, keepdims=True),
                            X.max(axis=0, keepdims=True)])
        centroid = X.mean(axis=0, keepdims=True)
        return Leaf(bounds, centroid, X.shape[0])
    most_variant = X.var(axis=0).argmax()
    feature = X[:, most_variant]
    thr = pivot(feature)
    left = X[feature < thr]
    right = X[feature > thr]
    return Node(
        pivot_value=thr, pivot_feature=most_variant,
        left=make_tree(left, leaf_size=leaf_size, pivot=pivot),
        right=make_tree(right, leaf_size=leaf_size, pivot=pivot),
    )


def get_leaves(tree: KDTree) -> List[Leaf]:
    """Extract leaves of the KDTree
    
    Parameters
    ==========
    tree : KDTree
        KDTree constructed on the data
        
    Returns
    =======
    leaves : list of Leaf
        All the leaves from the full depth of the tree
    """
    if isinstance(tree, Leaf):
        return [tree]
    return get_leaves(tree.left) + get_leaves(tree.right)


class KDTreeInitialization(Initialization):
    """Initializes k-means by picking extreme KDTree box"""
    def __init__(self, distance: str, leaf_size: Union[int, float] = 0.01):
        self.distance = distance
        self.leaf_size = leaf_size

    def __call__(self, data: Data, number_of_centroids: int) -> Centroids:
        """Generate initial centroids for k-means algorithm"""
        _validate(data, number_of_centroids)
        tree = make_tree(data, leaf_size=self.leaf_size, pivot=threshold_otsu)
        leaves = get_leaves(tree)
        box_centroids = np.vstack([l.centroid for l in leaves])
        box_weights = np.array([l.count for l in leaves])

        residuals = _find_residuals(box_centroids, box_weights)
        centroids = np.nan * np.zeros((number_of_centroids, data.shape[1]))
        centroids[0] = box_centroids[np.argmax(residuals)]

        distances = np.inf * np.ones((data.shape[0], ))
        for i in range(1, number_of_centroids):
            current_distance = dist.cdist(
                data, centroids[np.newaxis, i - 1], self.distance)
            distances[:] = np.minimum(current_distance.ravel(), distances)
            centroids[i] = data[np.argmax(distances)]

        return centroids
