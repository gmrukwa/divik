import logging
from typing import Tuple, Union

import dask.array as da
import dask.dataframe as dd
import dask_distance as ddst
import numpy as np
import scipy.spatial.distance as dst
from sklearn.base import (
    BaseEstimator,
    ClusterMixin,
    TransformerMixin,
)
from sklearn.utils.validation import check_is_fitted

from divik.cluster._kmeans._initialization import (
    ExtremeInitialization,
    Initialization,
    KDTreeInitialization,
    KDTreePercentileInitialization,
    PercentileInitialization,
)
from divik.core import (
    Centroids,
    Data,
    IntLabels,
    SegmentationMethod,
    configurable,
    normalize_rows,
)


class Labeling(object):
    """Labels observations by closest centroids"""

    def __init__(self, distance_metric: str, allow_dask: bool = False):
        """
        @param distance_metric: distance metric for estimation of closest
        @param allow_dask: should be False if `multiprocessing.Pool` is spawned
        """
        self.distance_metric = distance_metric
        self.allow_dask = allow_dask

    def __call__(self, data: Data, centroids: Centroids) -> IntLabels:
        """Find closest centroids

        @param data: observations in rows
        @param centroids: centroids in rows
        @return: vector of labels of centroids closest to points
        """
        if data.shape[1] != centroids.shape[1]:
            msg = (
                "Dimensionality of data and centroids must be equal. "
                + f"Was {data.shape[1]} and {centroids.shape[1]}"
            )
            logging.error(msg)
            raise ValueError(msg)

        if self.allow_dask and (data.shape[0] > 10000 or data.shape[1] > 1000):
            X1 = da.from_array(data)
            X2 = da.from_array(centroids)
            distances = ddst.cdist(X1, X2, self.distance_metric)
            labels = da.argmin(distances, axis=1).compute()
        else:
            distances = dst.cdist(data, centroids, self.distance_metric)
            labels = np.argmin(distances, axis=1)
        return labels


def redefine_centroids(
    data: Data, labeling: IntLabels, label_set: IntLabels, allow_dask: bool = False
) -> Centroids:
    """Recompute centroids in data for given labeling

    @param data: observations
    @param labeling: partition of dataset into groups
    @param label_set: set of labels used for partitioning
    @param allow_dask: should be False if `multiprocessing.Pool` is spawned
    @return: centroids
    """
    if data.shape[0] != labeling.size:
        msg = (
            "Each observation must have label specified. Number "
            + f"of labels: {labeling.size}, "
            + f"number of observations: {data.shape[0]}."
        )
        logging.error(msg)
        raise ValueError(msg)
    if allow_dask and (data.shape[0] > 10000 or data.shape[1] > 1000):
        X = dd.from_array(data)
        y = dd.from_array(labeling)
        centroids = X.groupby(y).mean().compute().values
    else:
        centroids = np.nan * np.zeros((len(label_set), data.shape[1]))
        for label in label_set:
            centroids[label] = np.mean(data[labeling == label], axis=0)
    return centroids


def _validate_kmeans_input(data: Data, number_of_clusters: int):
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        logging.error("data is expected to be 2D np.array")
        raise ValueError("data is expected to be 2D np.array")
    if number_of_clusters < 1:
        msg = "number_of_clusters({0}) < 1".format(number_of_clusters)
        logging.error(msg)
        raise ValueError(msg)


def _validate_normalizable(data):
    is_constant = data.min(axis=1) == data.max(axis=1)
    if is_constant.any():
        constant_rows = np.where(is_constant)[0]
        msg = "Constant rows {0} are not allowed for normalization."
        logging.error(msg.format(constant_rows))
        raise ValueError(msg.format(constant_rows))


class _KMeans(SegmentationMethod):
    """K-means clustering"""

    def __init__(
        self,
        labeling: Labeling,
        initialize: Initialization,
        number_of_iterations: int = 100,
        normalize_rows: bool = False,
        allow_dask: bool = False,
    ):
        """
        @param labeling: labeling method
        @param initialize: initialization method
        @param number_of_iterations: number of iterations
        @param normalize_rows: sets mean of row to 0 and norm to 1
        @param allow_dask: should be False if `multiprocessing.Pool` is spawned
        """
        self.labeling = labeling
        self.initialize = initialize
        self.number_of_iterations = number_of_iterations
        self.normalize_rows = normalize_rows
        self.allow_dask = allow_dask

    def _fix_labels(self, data, centroids, labels, n_clusters, retries=10):
        logging.debug("A label vanished - fixing")
        new_labels = labels.copy()
        known_labels = np.unique(labels)
        expected_labels = np.arange(n_clusters)
        missing_labels = np.setdiff1d(expected_labels, known_labels)
        logging.debug(
            "Missing labels ({0} were expected): {1}".format(n_clusters, missing_labels)
        )
        new_centroids = np.nan * np.zeros((n_clusters, centroids.shape[1]))
        for known in known_labels:
            new_centroids[known] = centroids[known]
        for missing in missing_labels:
            logging.debug("Fixing label: {0}".format(missing))
            new_center = np.nanmin(
                dst.cdist(data, new_centroids, metric=self.labeling.distance_metric),
                axis=1,
            ).argmax()
            logging.debug("Assigning to label: {0}".format(labels[new_center]))
            new_labels[new_center] = missing
            new_centroids[missing] = data[new_center]
        if np.unique(new_labels).size != n_clusters and retries > 0:
            logging.debug("fixed but lost another: {0}".format(np.unique(new_labels)))
            return self._fix_labels(
                data, new_centroids, new_labels, n_clusters, retries - 1
            )
        return new_centroids, new_labels

    def __call__(
        self, data: Data, number_of_clusters: int
    ) -> Tuple[IntLabels, Centroids]:
        _validate_kmeans_input(data, number_of_clusters)
        if number_of_clusters == 1:
            return (
                np.zeros((data.shape[0], 1), dtype=int),
                np.mean(data, axis=0, keepdims=True),
            )
        data = data.reshape(data.shape, order="C")
        if self.normalize_rows:
            _validate_normalizable(data)
            data = normalize_rows(data)
        label_set = np.arange(number_of_clusters)
        logging.debug("Initializing KMeans centroids.")
        centroids = self.initialize(data, number_of_clusters)
        logging.debug("First centroids found.")
        old_labels = np.nan * np.zeros((data.shape[0],))
        labels = self.labeling(data, centroids)
        logging.debug("Labels assigned.")
        for _ in range(self.number_of_iterations):
            if np.unique(labels).size != number_of_clusters:
                centroids, labels = self._fix_labels(
                    data, centroids, labels, number_of_clusters
                )
            if np.all(labels == old_labels):
                logging.debug("Stability achieved.")
                break
            old_labels = labels
            centroids = redefine_centroids(data, old_labels, label_set, self.allow_dask)
            labels = self.labeling(data, centroids)
        return labels, centroids


def _parse_initialization(
    name: str,
    distance: str,
    percentile: float = None,
    leaf_size: Union[int, float] = 0.01,
) -> Initialization:
    if name == "percentile":
        return PercentileInitialization(distance, percentile)
    if name == "extreme":
        return ExtremeInitialization(distance)
    if name == "kdtree":
        return KDTreeInitialization(distance, leaf_size)
    if name == "kdtree_percentile":
        return KDTreePercentileInitialization(distance, leaf_size, percentile)
    logging.error("Unknown initialization: {0}".format(name))
    raise ValueError("Unknown initialization: {0}".format(name))


@configurable
class KMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Means clustering

    Parameters
    ----------

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    distance : str, optional, default: 'euclidean'
        Distance measure. One of the distances supported by scipy package.

    init : {'percentile', 'extreme', 'kdtree', 'kdtree_percentile'}
        Method for initialization, defaults to 'percentile':

        'percentile' : selects initial cluster centers for k-mean
        clustering starting from specified percentile of distance to
        already selected clusters

        'extreme': selects initial cluster centers for k-mean
        clustering starting from the furthest points to already specified
        clusters

        'kdtree': selects initial cluster centers for k-mean clustering
        starting from centroids of KD-Tree boxes

        'kdtree_percentile': selects initial cluster centers for k-means
        clustering starting from centroids of KD-Tree boxes containing
        specified percentile. This should be more robust against outliers.

    percentile : float, default: 95.0
        Specifies the starting percentile for 'percentile' initialization.
        Must be within range [0.0, 100.0]. At 100.0 it is equivalent to
        'extreme' initialization.

    leaf_size : int or float, optional (default 0.01)
        Desired leaf size in kdtree initialization. When int, the box size
        will be between `leaf_size` and `2 * leaf_size`. When float, it will
        be between `leaf_size * n_samples` and `2 * leaf_size * n_samples`

    max_iter : int, default: 100
        Maximum number of iterations of the k-means algorithm for a
        single run.

    normalize_rows : bool, default: False
        If True, rows are translated to mean of 0.0 and scaled to norm of 1.0.

    allow_dask : bool, default: False
        If True, automatically selects dask as computations backend whenever
        reasonable. Default `False` since it cannot be used together with
        `multiprocessing.Pool` and everywhere `n_jobs` must be set to `1`.

    Attributes
    ----------

    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.

    labels_ :
        Labels of each point

    """

    # TODO: Add example of usage.
    def __init__(
        self,
        n_clusters: int,
        distance: str = "euclidean",
        init: str = "percentile",
        percentile: float = 95.0,
        leaf_size: Union[int, float] = 0.01,
        max_iter: int = 100,
        normalize_rows: bool = False,
        allow_dask: bool = False,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.distance = distance
        self.init = init
        self.percentile = percentile
        self.leaf_size = leaf_size
        self.max_iter = max_iter
        self.normalize_rows = normalize_rows
        self.allow_dask = allow_dask

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
            self.init, self.distance, self.percentile, self.leaf_size
        )
        kmeans = _KMeans(
            labeling=Labeling(self.distance, allow_dask=self.allow_dask),
            initialize=initialize,
            number_of_iterations=self.max_iter,
            normalize_rows=self.normalize_rows,
            allow_dask=self.allow_dask,
        )
        X = np.asanyarray(X)
        self.labels_, self.cluster_centers_ = kmeans(
            X, number_of_clusters=self.n_clusters
        )
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
        labels = dst.cdist(X, self.cluster_centers_, self.distance).argmin(axis=1)
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
