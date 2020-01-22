from typing import Union

import numpy as np

from ._base import DiviKBase
from ._dunn import dunn_divik
from ._report import DivikReporter
from divik.cluster import _kmeans as km


class DunnDiviK(DiviKBase):
    """DiviK clustering

    Parameters
    ----------

    gap_trials: int, optional, default: 10
        The number of random dataset draws to estimate the GAP index for the
        clustering quality assessment.

    leaf_size : int or float, optional (default 0.01)
        Desired leaf size in kdtree initialization. When int, the box size
        will be between `leaf_size` and `2 * leaf_size`. When float, it will
        be between `leaf_size * n_samples` and `2 * leaf_size * n_samples`

    max_iter: int, optional, default: 100
        Maximum number of iterations of the k-means algorithm for a single run.

    distance: str, optional, default: 'correlation'
        The distance metric between points, centroids and for GAP index
        estimation. One of the distances supported by scipy package.

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

    features_percentage: float, optional, default: 0.05
        The target percentage of features that are used by fallback percentage
        filter for 'outlier' filter.

    fast_kmeans_iter: int, optional, default: 10
        Maximum number of iterations of the k-means algorithm for a single run
        during computation of the GAP index. Decreased with respect to the
        max_iter, as GAP index requires multiple segmentations to be evaluated.

    k_max: int, optional, default: 10
        Maximum number of clusters evaluated during the auto-tuning process.
        From 1 up to k_max clusters are tested per evaluation.

    sample_size: int, optional, default: 10000
        Size of the sample used for GPA statistic computation.


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

    random_seed: int, optional, default: 0
        Seed to initialize the random number generator.

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

    >>> from divik.cluster import DunnDiviK
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=200, n_features=100, centers=20,
    ...                   random_state=42)
    >>> divik = DunnDiviK(distance='euclidean').fit(X)
    >>> divik.labels_
    array([1, 1, 1, 0, ..., 0, 0], dtype=int32)
    >>> divik.predict([[0, ..., 0], [12, ..., 3]])
    array([1, 0], dtype=int32)
    >>> divik.cluster_centers_
    array([[10., ...,  2.],
           ...,
           [ 1, ...,  2.]])

    """
    def __init__(self,
                 gap_trials: int = 10,
                 leaf_size: Union[int, float] = 0.01,
                 max_iter: int = 100,
                 distance: str = 'correlation',
                 minimal_size: int = None,
                 rejection_size: int = None,
                 rejection_percentage: float = None,
                 minimal_features_percentage: float = .01,
                 features_percentage: float = 0.05,
                 fast_kmeans_iter: int = 10,
                 k_max: int = 10,
                 sample_size: int = 10000,
                 normalize_rows: bool = None,
                 use_logfilters: bool = False,
                 filter_type='gmm',
                 n_jobs: int = None,
                 random_seed: int = 0,  # TODO: Rework to use RandomState
                 verbose: bool = False):
        super().__init__(gap_trials, leaf_size, max_iter, distance,
                         minimal_size, rejection_size, rejection_percentage,
                         minimal_features_percentage, features_percentage,
                         k_max, sample_size, normalize_rows, use_logfilters,
                         filter_type, n_jobs, random_seed, verbose)
        self.fast_kmeans_iter = fast_kmeans_iter
        self._validate_arguments()
        if self.fast_kmeans_iter > self.max_iter or self.fast_kmeans_iter < 0:
            raise ValueError('fast_kmeans_iter must be in range [0, max_iter]')

    def _fast_kmeans(self):
        single_kmeans = km.KMeans(
            n_clusters=2, distance=self.distance, init='kdtree',
            leaf_size=self.leaf_size,
            max_iter=self.fast_kmeans_iter,
            normalize_rows=self._needs_normalization())
        kmeans = km.GAPSearch(
            single_kmeans, max_clusters=2, n_jobs=self.n_jobs,
            seed=self.random_seed, n_trials=self.gap_trials,
            sample_size=self.sample_size, verbose=self.verbose)
        return kmeans

    def _full_kmeans(self):
        single_kmeans = km.KMeans(
            n_clusters=2, distance=self.distance, init='kdtree',
            leaf_size=self.leaf_size,
            max_iter=self.max_iter,
            normalize_rows=self._needs_normalization())
        kmeans = km.DunnSearch(
            single_kmeans, max_clusters=self.k_max, n_jobs=self.n_jobs,
            verbose=self.verbose)
        return kmeans

    def _divik(self, X, progress):
        fast = self._fast_kmeans()
        full = self._full_kmeans()
        warn_const = fast.kmeans.normalize_rows or full.kmeans.normalize_rows
        report = DivikReporter(progress, warn_const=warn_const)
        select_all = np.ones(shape=(X.shape[0],), dtype=bool)
        minimal_size = int(X.shape[0] * 0.001) if self.minimal_size is None \
            else self.minimal_size
        rejection_size = self._get_rejection_size(X)
        return dunn_divik(
            X, selection=select_all, fast_kmeans=fast,
            full_kmeans=full,
            feature_selector=self._feature_selector(X.shape[1]),
            minimal_size=minimal_size, rejection_size=rejection_size,
            report=report)
