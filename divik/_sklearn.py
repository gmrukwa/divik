from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import tqdm

import divik._divik as dv
import divik._distance as dst
import divik._feature_selection as fs
import divik._kmeans as km
import divik._summary as summary
from divik._utils import normalize_rows, DivikResult, get_n_jobs


class DiviK(BaseEstimator, ClusterMixin, TransformerMixin):
    """DiviK clustering

    Parameters
    ----------

    gap_trials: int, optional, default: 10
        The number of random dataset draws to estimate the GAP index for the
        clustering quality assessment.

    distance_percentile: float, optional, default: 99.0
        The percentile of the distance between points and their closest
        centroid. 100.0 would simply select the furthest point from all the
        centroids found already. Lower value provides better robustness against
        outliers. Too low value reduces the capability to detect centroid
        candidates during initialization.

    max_iter: int, optional, default: 100
        Maximum number of iterations of the k-means algorithm for a single run.

    distance: str, optional, default: 'correlation'
        The distance metric between points, centroids and for GAP index
        estimation.

    minimal_size: int, optional, default: None
        The minimum size of the region (the number of observations) to be
        considered for any further divisions. When left None, defaults to
        0.1% of the training dataset size.

    rejection_size: int, optional, default: None
        Size under which split will be rejected - if a cluster appears in the
        split that is below rejection_size, the split is considered improper
        and discarded. This may be useful for some domains (like there is no
        justification for a 3-cells cluster in biological data). By default,
        no segmentation is discarded, as careful post-processing provides the
        same advantage.

    rejection_percentage: float, optional, default: None
        An alternative to ``rejection_size``, with the same behavior, but this
        parameter is related to the training data size percentage. By default,
        no segmentation is discarded.

    minimal_features_percentage: float, optional, default: 0.01
        The minimal percentage of features that must be preserved after
        GMM-based feature selection. By default at least 1% of features is
        preserved in the filtration process.

    fast_kmeans_iter: int, optional, default: 10
        Maximum number of iterations of the k-means algorithm for a single run
        during computation of the GAP index. Decreased with respect to the
        max_iter, as GAP index requires multiple segmentations to be evaluated.

    k_max: int, optional, default: 10
        Maximum number of clusters evaluated during the auto-tuning process.
        From 1 up to k_max clusters are tested per evaluation.

    normalize_rows: bool, optional, default: None
        Whether to normalize each row of the data to the norm of 1. By default,
        it normalizes rows for correlation metric, does no normalization
        otherwise.

    use_logfilters: bool, optional, default: False
        Whether to compute logarithm of feature characteristic instead of the
        characteristic itself. This may improve feature filtering performance,
        depending on the distribution of features, however all the
        characteristics (mean, variance) have to be positive for that -
        filtering will fail otherwise. This is useful for specific cases in
        biology where the distribution of data may actually require this option
        for any efficient filtering.

    n_jobs : int, optional, default: None
        The number of jobs to use for the computation. This works by computing
        each of the GAP index evaluations in parallel and by making predictions
        in parallel.

    random_seed: int, optional, default: 0
        Seed to initialize the random number generator.

    verbose : bool, optional, default: False
        Whether to report the progress of the computations.

    Attributes
    ----------

    result_ : divik.types.DivikResult
        Hierarchical structure describing all the consecutive segmentations.

    labels_ :
        Labels of each point

    centroids_ : array, [n_clusters, n_features]
        Coordinates of cluster centers. If the algorithm stops before fully
        converging, these will not be consistent with ``labels_``. Also, the
        distance between points and respective centroids must be captured
        in appropriate features subspace. This is realized by the ``transform``
        method.

    filters_ : array, [n_clusters, n_features]
        Filters that were applied to the feature space on the level that was
        the final segmentation for a subset.

    depth_ : int
        The number of hierarchy levels in the segmentation.

    n_clusters_ : int
        The final number of clusters in the segmentation, on the tree leaf
        level.

    paths_ : Dict[int, Tuple[int]]
        Describes how the cluster number corresponds to the path in the tree.
        Element of the tuple indicates the sub-segment number on each tree
        level.

    reverse_paths_ : Dict[Tuple[int], int]
        Describes how the path in the tree corresponds to the cluster number.
        For more details see ``paths_``.

    Examples
    --------

    >>> from divik import DiviK
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=200, n_features=100, centers=20,
    ...                   random_state=42)
    >>> divik = DiviK(distance='euclidean').fit(X)
    >>> divik.labels_
    array([1, 1, 1, 0, ..., 0, 0], dtype=int32)
    >>> divik.predict([[0, ..., 0], [12, ..., 3]])
    array([1, 0], dtype=int32)
    >>> divik.cluster_centers_
    array([[10., ...,  2.],
           ...,
           [ 1, ...,  2.]])

    """
    # TODO: Improve docstring so it will work with doctest

    def __init__(self,
                 gap_trials: int = 10,
                 distance_percentile: float = 99.,
                 max_iter: int = 100,
                 distance: str = dst.KnownMetric.correlation.value,
                 minimal_size: int = None,
                 rejection_size: int = None,
                 rejection_percentage: float = None,
                 minimal_features_percentage: float = .01,
                 fast_kmeans_iter: int = 10,
                 k_max: int = 10,
                 normalize_rows: bool = None,
                 use_logfilters: bool = False,
                 n_jobs: int = None,
                 random_seed: int = 0,  # TODO: Rework to use RandomState
                 verbose: bool = False):
        self.gap_trials = gap_trials
        self.distance_percentile = distance_percentile
        self.max_iter = max_iter
        self.distance = distance
        self.minimal_size = minimal_size
        self.rejection_size = rejection_size
        self.rejection_percentage = rejection_percentage
        self.minimal_features_percentage = minimal_features_percentage
        self.fast_kmeans_iter = fast_kmeans_iter
        self.k_max = k_max
        self.normalize_rows = normalize_rows
        self.use_logfilters = use_logfilters
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.verbose = verbose
        self._validate_arguments()

    def _validate_arguments(self):
        if self.distance not in list(dst.KnownMetric):
            raise ValueError('Unknown distance: %s' % self.distance)
        if self.gap_trials <= 0:
            raise ValueError('gap_trials must be greater than 0')
        if self.distance_percentile < 0 or self.distance_percentile > 100:
            raise ValueError('distance_percentile must be in range [0, 100]')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be greater than 0')
        if self.minimal_size is not None and self.minimal_size < 0:
            raise ValueError('minimal_size must be greater or equal to 0')
        if self.minimal_features_percentage < 0 \
                or self.minimal_features_percentage > 1:
            raise ValueError('minimal_features_percentage must be in range'
                             ' [0, 1]')
        if self.fast_kmeans_iter > self.max_iter or self.fast_kmeans_iter < 0:
            raise ValueError('fast_kmeans_iter must be in range [0, max_iter]')

    def fit(self, X, y=None):
        """Compute DiviK clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        y : Ignored
            not used, present here for API consistency by convention.
        """
        if np.isnan(X).any():
            raise ValueError("NaN values are not supported.")
        minimal_size = int(X.shape[0] * 0.001) if self.minimal_size is None \
            else self.minimal_size
        rejection_size = self._get_rejection_size(X)
        n_jobs = get_n_jobs(self.n_jobs)

        with context_if(self.verbose, tqdm.tqdm, total=X.shape[0]) as progress,\
             context_if(n_jobs != 1, Pool, n_jobs) as pool:
            self.result_ = dv.divik(
                X, fast_kmeans=self._fast_kmeans(),
                full_kmeans=self._full_kmeans(),
                feature_selector=self._feature_selector(),
                progress_reporter=progress, minimal_size=minimal_size,
                rejection_size=rejection_size, pool=pool)

        self.labels_, self.paths_ = summary.merged_partition(self.result_,
                                                             return_paths=True)
        self.reverse_paths_ = {value: key for key, value in self.paths_.items()}
        self.filters_ = np.array(
            [self._get_filter(path) for path in self.reverse_paths_],
            dtype=bool)
        self.centroids_ = pd.DataFrame(X).groupby(self.labels_, sort=True)\
            .mean().values
        self.depth_ = summary.depth(self.result_)
        self.n_clusters_ = summary.total_number_of_clusters(self.result_)

        return self

    def _get_rejection_size(self, X):
        rejection_size = 0
        if self.rejection_size is not None:
            rejection_size = max(rejection_size, self.rejection_size)
        if self.rejection_percentage is not None:
            rejection_size = max(
                rejection_size, int(self.rejection_percentage * X.shape[0]))
        return rejection_size

    def _get_filter(self, path):
        """This method extracts features filter used for each centroid"""
        result = self.result_
        for item in path[:-1]:
            result = result.subregions[item]
        return result.feature_selector.selected_

    def _needs_normalization(self):
        if self.normalize_rows is None:
            return self.distance == dst.KnownMetric.correlation.value
        return self.normalize_rows

    def _fast_kmeans(self):
        return km.AutoKMeans(
            max_clusters=2, n_jobs=get_n_jobs(self.n_jobs), method="gap",
            distance=self.distance, init='percentile',
            percentile=self.distance_percentile, max_iter=self.max_iter,
            normalize_rows=self._needs_normalization(),
            gap={"max_iter": self.fast_kmeans_iter, "seed": self.random_seed,
                 "trials": self.gap_trials, "correction": True},
            verbose=self.verbose)

    def _full_kmeans(self):
        return km.AutoKMeans(
            max_clusters=self.k_max, min_clusters=2,
            n_jobs=get_n_jobs(self.n_jobs), method='dunn',
            distance=self.distance, init='percentile',
            percentile=self.distance_percentile, max_iter=self.max_iter,
            normalize_rows=self._needs_normalization(), gap=None,
            verbose=self.verbose
        )

    def _feature_selector(self):
        return fs.HighAbundanceAndVarianceSelector(
            use_log=self.use_logfilters,
            min_features_rate=self.minimal_features_percentage)

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        y : Ignored
            not used, present here for API consistency by convention.

        Returns
        -------

        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_

    def fit_transform(self, X, y=None, **fit_params):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        y : Ignored
            not used, present here for API consistency by convention.

        Returns
        -------

        X_new : array, shape [n_samples, n_clusters_]
            X transformed in the new space.
        """
        return self.fit(X).transform(X)

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

        X_new : array, shape [n_samples, n_clusters_]
            X transformed in the new space.
        """
        if self._needs_normalization():
            X = normalize_rows(X)
        distance = dst.ScipyDistance(dst.KnownMetric[self.distance])
        distances = np.hstack([
            distance(X[:, selector], centroid[np.newaxis, selector])
            for selector, centroid in zip(self.filters_, self.centroids_)
        ])
        return distances

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
        if self._needs_normalization():
            X = normalize_rows(X)
        distance = dst.ScipyDistance(dst.KnownMetric[self.distance])
        n_jobs = get_n_jobs(self.n_jobs)
        predict = partial(_predict_path, result=self.result_, distance=distance)
        if n_jobs == 1:
            paths = [_predict_path(row, self.result_, distance) for row in X]
        else:
            with Pool(n_jobs) as pool:
                paths = pool.map(predict, X)
        labels = [self.reverse_paths_[path] for path in paths]
        return np.array(labels, dtype=np.int32)


def _predict_path(observation: np.ndarray, result: DivikResult, distance) \
        -> Tuple[int]:
    path = []
    observation = observation[np.newaxis, :]
    division = result
    while division is not None:
        local_X = division.feature_selector.transform(observation)
        label = int(division.clustering.predict(local_X))
        path.append(label)
        division = division.subregions[label]
    path = tuple(path)
    return path


@contextmanager
def context_if(condition, context, *args, **kwargs):
    if condition:
        with context(*args, **kwargs) as c:
            yield c
    else:
        yield None
