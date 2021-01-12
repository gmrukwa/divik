import logging
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import (
    List,
    NamedTuple,
    Union,
)

import numpy as np
import scipy.spatial.distance as dist
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
        raise NotImplementedError(self.__class__.__name__ + " must implement __call__.")


def _find_residuals(data: Data, sample_weight=None) -> np.ndarray:
    features = data.T
    assumed_ys = features[0]
    modelled_xs = np.hstack([np.ones((data.shape[0], 1)), features[1:].T])
    lr = LinearRegression().fit(modelled_xs, assumed_ys, sample_weight=sample_weight)
    residuals = np.abs(lr.predict(modelled_xs) - assumed_ys)
    return residuals


def _validate(data: Data, number_of_centroids: int):
    if number_of_centroids > data.shape[0]:
        msg = (
            f"Number of centroids ({number_of_centroids}) greater than "
            + f"number of observations ({data.shape[0]})"
        )
        logging.error(msg)
        raise ValueError(msg)
    if number_of_centroids < 1:
        msg = f"number_of_centroids({number_of_centroids}) < 1"
        logging.error(msg)
        raise ValueError(msg)


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

        distances = np.inf * np.ones((data.shape[0],))
        for i in range(1, number_of_centroids):
            current_distance = dist.cdist(
                data, centroids[np.newaxis, i - 1], self.distance
            )
            distances[:] = np.minimum(current_distance.ravel(), distances)
            centroids[i] = data[np.argmax(distances)]

        return centroids


class PercentileInitialization(Initialization):
    def __init__(self, distance: str, percentile: float = 99.0):
        assert 0 <= percentile <= 100, percentile
        self.distance = distance
        self.percentile = percentile

    def _get_percentile_element(self, values: np.ndarray) -> int:
        value = np.percentile(values, q=self.percentile, interpolation="nearest")
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
                data, centroids[np.newaxis, i - 1], self.distance
            )
            nans = np.isnan(current_distance)
            if np.any(nans):
                locations_of_nans = np.array(list(zip(*np.nonzero(nans))))
                msg = (
                    "Distances between points cannot be NaN. This "
                    + "indicates that your data is probably corrupted and "
                    + "analysis cannot be continued in this setting. Amount"
                    + f" of NaNs: {nans.sum()}. At positions described by "
                    + f"[spot, centroid]: {locations_of_nans}"
                )
                logging.error(msg)
                raise ValueError(msg)
            distances[:] = np.minimum(current_distance.ravel(), distances)
            selected = self._get_percentile_element(distances)
            centroids[i] = data[selected]

        return centroids


class Leaf(NamedTuple):
    centroid: np.ndarray
    count: int = 0


KDTree = Union["Node", Leaf]


class Node(NamedTuple):
    left: KDTree = None
    right: KDTree = None


def make_tree(X, leaf_size: int, _feature_idx: int = 0, selector=None) -> KDTree:
    """Make KDTree out of the data

    Construct a KDTree out of data using mean as a pivoting element.
    Each split makes two segments. The result doesn't contain the original
    data, just centroids in each box and count of items.

    Parameters
    ==========
    X : array_like, (n_samples, n_features)
        Set of observations to divide into boxes

    leaf_size : int
        Desired leaf size. It should more than `leaf_size` and
        will be up to `2 * leaf_size`

    Returns
    =======
    tree : KDTree
        Lightweight KD-Tree over the data
    """
    if selector is None:
        selector = np.ones((X.shape[0],), dtype=bool)
    if selector.sum() < 2 * leaf_size:
        centroid = X[selector, :].mean(axis=0, keepdims=True)
        return Leaf(centroid, int(selector.sum()))
    feature = X[selector, _feature_idx]
    thr = np.mean(feature)
    left_idx = selector.copy()
    left_idx[selector] = feature < thr
    right_idx = selector.copy()
    right_idx[selector] = np.logical_not(left_idx[selector])
    if left_idx[selector].all() or right_idx[selector].all():
        centroid = X[selector, :].mean(axis=0, keepdims=True)
        return Leaf(centroid, int(selector.sum()))
    next_feature = (_feature_idx + 1) % X.shape[1]
    return Node(
        left=make_tree(
            X, leaf_size=leaf_size, _feature_idx=next_feature, selector=left_idx
        ),
        right=make_tree(
            X, leaf_size=leaf_size, _feature_idx=next_feature, selector=right_idx
        ),
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
        leaf_size = self.leaf_size
        if isinstance(leaf_size, float):
            if 0 <= leaf_size <= 1:
                leaf_size = max(int(leaf_size * data.shape[0]), 1)
            else:
                logging.error("leaf_size must be between 0 and 1 when float")
                raise ValueError("leaf_size must be between 0 and 1 when float")
        tree = make_tree(data, leaf_size=leaf_size)
        leaves = get_leaves(tree)
        box_centroids = np.vstack([l.centroid for l in leaves])
        box_weights = np.array([l.count for l in leaves])

        residuals = _find_residuals(box_centroids, box_weights)
        centroids = np.nan * np.zeros((number_of_centroids, data.shape[1]))
        centroids[0] = box_centroids[np.argmax(residuals)]

        distances = np.inf * np.ones((box_centroids.shape[0],))
        for i in range(1, number_of_centroids):
            current_distance = dist.cdist(
                box_centroids, centroids[np.newaxis, i - 1], self.distance
            )
            distances[:] = np.minimum(current_distance.ravel(), distances)
            centroids[i] = box_centroids[np.argmax(distances)]

        return centroids


class KDTreePercentileInitialization(Initialization):
    """Initializes k-means by picking extreme KDTree box"""

    def __init__(
        self,
        distance: str,
        leaf_size: Union[int, float] = 0.01,
        percentile: float = 99.0,
    ):
        assert 0 <= percentile <= 100, percentile
        self.distance = distance
        self.leaf_size = leaf_size
        self.percentile = percentile

    def _get_percentile_idx(self, distances, weights) -> int:
        idx = np.argsort(distances)
        over_percentile = np.cumsum(weights[idx]) >= self.percentile
        first_over = np.flatnonzero(over_percentile)[0]
        return idx[first_over]

    def __call__(self, data: Data, number_of_centroids: int) -> Centroids:
        """Generate initial centroids for k-means algorithm"""
        _validate(data, number_of_centroids)
        leaf_size = self.leaf_size
        if isinstance(leaf_size, float):
            if 0 <= leaf_size <= 1:
                leaf_size = max(int(leaf_size * data.shape[0]), 1)
            else:
                logging.error("leaf_size must be between 0 and 1 when float")
                raise ValueError("leaf_size must be between 0 and 1 when float")
        tree = make_tree(data, leaf_size=leaf_size)
        leaves = get_leaves(tree)
        box_centroids = np.vstack([l.centroid for l in leaves])
        box_weights = np.array([l.count for l in leaves])
        normalized_weights = 100 * box_weights / np.sum(box_weights)

        residuals = _find_residuals(box_centroids, box_weights)
        centroids = np.nan * np.zeros((number_of_centroids, data.shape[1]))
        idx = self._get_percentile_idx(residuals, normalized_weights)
        centroids[0] = box_centroids[idx]

        distances = np.inf * np.ones((box_centroids.shape[0],))
        for i in range(1, number_of_centroids):
            current_distance = dist.cdist(
                box_centroids, centroids[np.newaxis, i - 1], self.distance
            )
            distances[:] = np.minimum(current_distance.ravel(), distances)
            idx = self._get_percentile_idx(distances, normalized_weights)
            centroids[i] = box_centroids[idx]

        return centroids
