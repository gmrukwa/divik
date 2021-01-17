import logging
import sys
import uuid
from functools import partial

import numpy as np
import tqdm
from sklearn.base import (
    BaseEstimator,
    ClusterMixin,
    TransformerMixin,
    clone,
)
from sklearn.utils.validation import check_is_fitted

from divik.core import (
    configurable,
    maybe_pool,
    share,
)
from divik.score import dunn, sampled_dunn

from ._core import KMeans

_DATA = {}


def _pool_initialize(ref, data):
    _DATA[ref] = data


@configurable
class DunnSearch(BaseEstimator, ClusterMixin, TransformerMixin):
    """Select best number of clusters for k-means

    Parameters
    ----------
    kmeans : KMeans
        KMeans object to tune

    max_clusters: int
        The maximal number of clusters to form and score.

    min_clusters: int, default: 1
        The minimal number of clusters to form and score.

    method: {'full', 'sampled', 'auto'}, default: 'full'
        Whether to run full computations or approximate.
        - full - always computes full Dunn's index, without sampling
        - sampled - samples the clusters to reduce computational overhead
        - auto - switches the above methods to provide best performance-quality
        trade-off.

    inter : {'centroid', 'closest'}, default: 'centroid'
        How the distance between clusters is computed. For more details see
        `dunn`.

    intra : {'avg', 'furthest'}, default: 'avg'
        How the cluster internal distance is computed. For more details see
        `dunn`.

    sample_size : int, default: 1000
        Size of the sample used to compute Dunn index in `auto` or `sampled`
        scenario.

    n_trials : int, default: 10
        Number of trials to use when computing Dunn index in `auto` or
        `sampled` scenario.

    seed : int, default: 42
        Random seed for the reproducibility of subset draws in Dunn `auto`
        or `sampled` scenario.

    n_jobs: int, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the clustering & scoring runs in parallel.

    drop_unfit: bool, default: False
        If True, drops the estimators that did not fit the data.

    verbose: bool, default: False
        If True, shows progress with tqdm.

    Attributes
    ----------
    cluster_centers_: array, [n_clusters, n_features]
        Coordinates of cluster centers.

    labels_:
        Labels of each point.

    estimators_: List[KMeans]
        KMeans instances for n_clusters in range [min_clusters, max_clusters].

    scores_: array, [max_clusters - min_clusters + 1,]
        Array with scores for each estimator.

    n_clusters_: int
        Estimated optimal number of clusters.

    best_score_: float
        Score of the optimal estimator.

    best_: KMeans
        The optimal estimator.

    """

    def __init__(
        self,
        kmeans: KMeans,
        max_clusters: int,
        min_clusters: int = 2,
        method="full",
        inter="centroid",
        intra="avg",
        sample_size=1000,
        n_trials=10,
        seed=42,
        n_jobs: int = 1,
        drop_unfit: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        assert min_clusters <= max_clusters
        self.kmeans = kmeans
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.method = method
        self.inter = inter
        self.intra = intra
        self.sample_size = sample_size
        self.n_trials = n_trials
        self.seed = seed
        self.n_jobs = n_jobs
        self.drop_unfit = drop_unfit
        self.verbose = verbose

    def _n_ops(self, data):
        if self.inter == "closest" or self.intra == "furthest":
            n_ops_full = data.shape[0] ** 2
            n_ops_sampled = self.n_trials * self.sample_size ** 2
        else:
            n_ops_full = data.shape[0]
            n_ops_sampled = self.n_trials * self.sample_size
        return n_ops_full, n_ops_sampled

    def _sampled_dunn(self, kmeans, data, inter, intra):
        return sampled_dunn(
            kmeans,
            data,
            inter=inter,
            intra=intra,
            sample_size=self.sample_size,
            n_jobs=1,
            seed=self.seed,
            n_trials=self.n_trials,
        )

    def _dunn(self, kmeans, data):
        n_ops_full, n_ops_sampled = self._n_ops(data)
        if self.method == "full":
            dunn_ = dunn
        elif self.method == "sampled":
            dunn_ = self._sampled_dunn
        elif self.method == "auto" and n_ops_full <= n_ops_sampled:
            dunn_ = dunn
            logging.debug('DunnSearch.method="auto" resolved to "full"')
        elif self.method == "auto" and n_ops_full > n_ops_sampled:
            dunn_ = self._sampled_dunn
            logging.debug('DunnSearch.method="auto" resolved to "sampled"')
        else:
            logging.error(f"Failed to select Dunn method {self.method}.")
            logging.debug(f"n_ops_full={n_ops_full}, " f"n_ops_sampled={n_ops_sampled}")
            raise ValueError(f"Unknown Dunn method {self.method}")
        return dunn_(kmeans, data, inter=self.inter, intra=self.intra)

    def _fit_kmeans(self, n_clusters, data_ref):
        data = _DATA[data_ref].value
        kmeans = clone(self.kmeans)
        kmeans.n_clusters = n_clusters
        kmeans.fit(data)
        d = self._dunn(kmeans, data)
        return kmeans, d

    def fit(self, X, y=None):
        """Compute k-means clustering and estimate optimal number of clusters.

        Parameters
        ----------

        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.

        y : Ignored
            not used, present here for API consistency by convention.

        """
        n_clusters = range(self.min_clusters, self.max_clusters + 1)
        if self.verbose:
            n_clusters = tqdm.tqdm(n_clusters, leave=False, file=sys.stdout)
        ref = str(uuid.uuid4())
        with share(X) as x:
            _DATA[ref] = x
            with maybe_pool(
                self.n_jobs, initializer=_pool_initialize, initargs=(ref, x)
            ) as pool:
                fit_kmeans = partial(self._fit_kmeans, data_ref=ref)
                kmeans_and_scores = pool.map(fit_kmeans, n_clusters)
            del _DATA[ref]
        logging.debug("Fitted DunnSearch")

        self.estimators_, self.scores_ = zip(*kmeans_and_scores)
        self.scores_ = np.array(self.scores_)
        best = np.argmax(self.scores_)
        self.fitted_ = True
        self.n_clusters_ = best + self.min_clusters
        self.best_score_ = self.scores_[best]
        self.best_ = self.estimators_[best]
        self.labels_ = self.best_.labels_
        self.cluster_centers_ = self.best_.cluster_centers_
        if self.drop_unfit:
            self.estimators_ = None

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
        return self.best_.predict(X)

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
        return self.best_.transform(X)
