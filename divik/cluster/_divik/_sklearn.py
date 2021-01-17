import sys
from functools import partial
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
from scipy.spatial import distance as dist
from sklearn.base import (
    BaseEstimator,
    ClusterMixin,
    TransformerMixin,
)
from sklearn.utils.validation import check_is_fitted

from divik import _summary as summary
from divik import feature_selection as fs
from divik.core import (
    DivikResult,
    configurable,
    context_if,
    maybe_pool,
    normalize_rows,
    visualize,
)
from divik.core.io import saver

from ._backend import divik
from ._report import DivikReporter


@configurable
class DiviK(BaseEstimator, ClusterMixin, TransformerMixin):
    """DiviK clustering

    Parameters
    ----------
    kmeans: AutoKMeans
        A self-tuning KMeans estimator for the purpose of clustering

    fast_kmeans: GAPSearch, optional, default: None
        A self-tuning KMeans estimator for the purpose of stop condition
        check. If None, the `kmeans` parameter is assumed to be the
        `GAPSearch` instance.

    distance: str, optional, default: 'correlation'
        The distance metric between points, centroids and for GAP index
        estimation. One of the distances supported by scipy package.

    minimal_size: int or float, optional, default: None
        The minimum size of the region (the number of observations) to be
        considered for any further divisions. If provided number is between
        0 and 1, it is considered a rate of training dataset size. When left
        None, defaults to 0.1% of the training dataset size.

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

    features_percentage: float, optional, default: 0.05
        The target percentage of features that are used by fallback percentage
        filter for 'outlier' filter.

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

    filter_type: {'gmm', 'outlier', 'auto', 'none'}, default: 'gmm'
        - 'gmm' - usual Gaussian Mixture Model-based filtering, useful for high
        dimensional cases
        - 'outlier' - robust outlier detection-based filtering, useful for low
        dimensional cases. In the case of no outliers, percentage-based
        filtering is applied.
        - 'auto' - automatically selects between 'gmm' and 'outlier' based on
        the dimensionality. When more than 250 features are present, 'gmm'
        is chosen.
        - 'none' - feature selection is disabled

    n_jobs: int, optional, default: None
        The number of jobs to use for the computation. This works by computing
        each of the GAP index evaluations in parallel and by making predictions
        in parallel.

    verbose: bool, optional, default: False
        Whether to report the progress of the computations.

    Attributes
    ----------

    result_: divik.DivikResult
        Hierarchical structure describing all the consecutive segmentations.

    labels_:
        Labels of each point

    centroids_: array, [n_clusters, n_features]
        Coordinates of cluster centers. If the algorithm stops before fully
        converging, these will not be consistent with ``labels_``. Also, the
        distance between points and respective centroids must be captured
        in appropriate features subspace. This is realized by the ``transform``
        method.

    filters_: array, [n_clusters, n_features]
        Filters that were applied to the feature space on the level that was
        the final segmentation for a subset.

    depth_: int
        The number of hierarchy levels in the segmentation.

    n_clusters_: int
        The final number of clusters in the segmentation, on the tree leaf
        level.

    paths_: Dict[int, Tuple[int]]
        Describes how the cluster number corresponds to the path in the tree.
        Element of the tuple indicates the sub-segment number on each tree
        level.

    reverse_paths_: Dict[Tuple[int], int]
        Describes how the path in the tree corresponds to the cluster number.
        For more details see ``paths_``.

    Examples
    --------

    >>> from divik.cluster import DiviK
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

    def __init__(
        self,
        kmeans,
        fast_kmeans=None,
        distance: str = "correlation",
        # TODO: Allow percentage
        minimal_size: int = None,
        rejection_size: int = None,
        rejection_percentage: float = None,
        minimal_features_percentage: float = 0.01,
        features_percentage: float = 0.05,
        normalize_rows: bool = None,
        use_logfilters: bool = False,
        filter_type="gmm",
        n_jobs: int = None,
        verbose: bool = False,
    ):
        self.kmeans = kmeans
        self.fast_kmeans = fast_kmeans
        self.distance = distance
        self.minimal_size = minimal_size
        self.rejection_size = rejection_size
        self.rejection_percentage = rejection_percentage
        self.minimal_features_percentage = minimal_features_percentage
        self.features_percentage = features_percentage
        self.normalize_rows = normalize_rows
        self.use_logfilters = use_logfilters
        self.filter_type = filter_type
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._validate_arguments()

    def _validate_arguments(self):
        if self.minimal_features_percentage < 0 or self.minimal_features_percentage > 1:
            raise ValueError("minimal_features_percentage must be in range" " [0, 1]")
        if self.features_percentage < 0 or self.features_percentage > 1:
            raise ValueError("features_percentage must be in range [0, 1]")
        if self.features_percentage < self.minimal_features_percentage:
            raise ValueError(
                "features_percentage must be higher than or equal"
                " to minimal_features_percentage"
            )
        if self.filter_type not in ["gmm", "outlier", "auto", "none"]:
            raise ValueError(
                "filter_type must be in ['gmm', 'outlier', 'auto', 'none']"
            )

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

        with context_if(
            self.verbose, tqdm.tqdm, total=X.shape[0], file=sys.stdout, smoothing=0
        ) as progress:
            self.result_ = self._divik(X, progress)

        if self.result_ is None:
            self.labels_ = np.zeros((X.shape[0],), dtype=int)
            self.paths_ = {0: (0,)}
        else:
            self.labels_, self.paths_ = summary.merged_partition(
                self.result_, return_paths=True
            )

        self.reverse_paths_ = {value: key for key, value in self.paths_.items()}

        if self.result_ is None:
            self.filters_ = np.ones([1, X.shape[1]], dtype=bool)
        else:
            self.filters_ = np.array(
                [self._get_filter(path) for path in self.reverse_paths_], dtype=bool
            )
        self.centroids_ = pd.DataFrame(X).groupby(self.labels_, sort=True).mean().values
        self.depth_ = summary.depth(self.result_)
        self.n_clusters_ = summary.total_number_of_clusters(self.result_)

        return self

    def _get_rejection_size(self, X):
        rejection_size = 0
        if self.rejection_size is not None:
            rejection_size = max(rejection_size, self.rejection_size)
        if self.rejection_percentage is not None:
            rejection_size = max(
                rejection_size, int(self.rejection_percentage * X.shape[0])
            )
        return rejection_size

    def _get_filter(self, path):
        """This method extracts features filter used for each centroid"""
        result = self.result_
        for item in path[:-1]:
            result = result.subregions[item]
        return result.feature_selector.selected_

    def _needs_normalization(self):
        if self.normalize_rows is None:
            return self.distance == "correlation"
        return self.normalize_rows

    def _feature_selector(self, n_features):
        return fs.make_specialized_selector(
            self.filter_type,
            n_features,
            use_log=self.use_logfilters,
            min_features_rate=self.minimal_features_percentage,
            p=self.features_percentage,
        )

    def _divik(self, X, progress):
        full = self.kmeans
        fast = self.fast_kmeans
        if fast is None:
            warn_const = full.kmeans.normalize_rows
        else:
            warn_const = fast.kmeans.normalize_rows or full.kmeans.normalize_rows
        report = DivikReporter(progress, warn_const=warn_const)
        select_all = np.ones(shape=(X.shape[0],), dtype=bool)
        if self.minimal_size is None:
            minimal_size = int(X.shape[0] * 0.001)
        elif 0 < self.minimal_size < 1:
            minimal_size = int(X.shape[0] * self.minimal_size)
        else:
            minimal_size = self.minimal_size
        rejection_size = self._get_rejection_size(X)
        return divik(
            X,
            selection=select_all,
            kmeans=full,
            fast_kmeans=fast,
            feature_selector=self._feature_selector(X.shape[1]),
            minimal_size=minimal_size,
            rejection_size=rejection_size,
            report=report,
        )

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

        X_new : array, shape [n_samples, self.n_clusters_]
            X transformed in the new space.
        """
        check_is_fitted(self)
        if self._needs_normalization():
            X = normalize_rows(X)
        distances = np.hstack(
            [
                dist.cdist(
                    X[:, selector], centroid[np.newaxis, selector], self.distance
                )
                for selector, centroid in zip(self.filters_, self.centroids_)
            ]
        )
        return distances

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

        X_new : array, shape [n_samples, self.n_clusters_]
            X transformed in the new space.
        """
        return self.fit(X).transform(X)

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
        if self._needs_normalization():
            X = normalize_rows(X)
        predict = partial(_predict_path, result=self.result_)
        with maybe_pool(self.n_jobs) as pool:
            paths = pool.map(predict, X)
        labels = [self.reverse_paths_[path] for path in paths]
        return np.array(labels, dtype=np.int32)


def _predict_path(observation: np.ndarray, result: DivikResult) -> Tuple[int]:
    path = []
    observation = observation[np.newaxis, :]
    division = result
    while division is not None:
        local_X = division.feature_selector.transform(observation)
        label = int(division.clustering.predict(local_X))
        path.append(label)
        division = division.subregions[label]
    path = tuple(path) if len(path) else (0,)
    return path


def make_merged(result: Optional[DivikResult]) -> np.ndarray:
    depth = summary.depth(result)
    return np.hstack(
        [
            summary.merged_partition(result, limit + 1).reshape(-1, 1)
            for limit in range(depth)
        ]
    )


def save_merged(fname_fn, merged: np.ndarray, xy: np.ndarray = None):
    np.savetxt(fname_fn("partitions.csv"), merged, delimiter=", ", fmt="%i")
    np.save(fname_fn("partitions.npy"), merged)
    import skimage.io

    if xy is not None:
        for level in range(merged.shape[1]):
            np.save(fname_fn("partitions.{0}.npy".format(level)), merged[:, level])
            visualization = visualize(merged[:, level], xy=xy)
            image_name = fname_fn("partitions.{0}.png".format(level))
            skimage.io.imsave(image_name, visualization)


@saver
def save_divik(model, fname_fn, **kwargs):
    if not hasattr(model, "result_"):
        return
    import logging

    if not isinstance(model.result_, DivikResult):
        logging.info("Skipping DiviK details save. Cause: result is None")
        return
    logging.info("Saving DiviK partitions.")
    merged = make_merged(model.result_).astype(np.int64)
    assert merged.shape[0] == model.result_.clustering.labels_.size
    xy = kwargs.get("xy", None)
    save_merged(fname_fn, merged, xy)
