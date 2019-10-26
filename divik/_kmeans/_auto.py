from functools import partial
from multiprocessing import Pool
import sys

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from divik._kmeans._core import KMeans
from divik._score import make_picker
from divik._utils import get_n_jobs


def _fit_kmeans(*args, **kwargs):
    data = kwargs.pop('data')
    return KMeans(*args, **kwargs).fit(data)


class AutoKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Means clustering with automated selection of number of clusters

    Parameters
    ----------

    max_clusters : int
        The maximal number of clusters to form and score.

    min_clusters : int, default: 1
        The minimal number of clusters to form and score.

    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the clustering & scoring runs in parallel.

    method: {'dunn', 'gap'}
        The method to select the best number of clusters.

        'dunn' : computes score that relates dispersion inside a cluster
        to distances between clusters. Never selects 1 cluster.

        'gap' : compares dispersion of a clustering to a dispersion in
        grouping of a reference uniformly distributed dataset

    distance : {'braycurtis', 'canberra', 'chebyshev', 'cityblock',
    'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
    'kulsinski', 'mahalanobis', 'atching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
        Distance measure, defaults to 'euclidean'. These are the distances
        supported by scipy package.

    init : {'percentile' or 'extreme'}
        Method for initialization, defaults to 'percentile':

        'percentile' : selects initial cluster centers for k-mean
        clustering starting from specified percentile of distance to
        already selected clusters

        'extreme': selects initial cluster centers for k-mean
        clustering starting from the furthest points to already specified
        clusters

    percentile : float, default: 95.0
        Specifies the starting percentile for 'percentile' initialization.
        Must be within range [0.0, 100.0]. At 100.0 it is equivalent to
        'extreme' initialization.

    max_iter : int, default: 100
        Maximum number of iterations of the k-means algorithm for a
        single run.

    normalize_rows : bool, default: False
        If True, rows are translated to mean of 0.0 and scaled to norm of 1.0.

    gap : dict
        Configuration of GAP statistic in a form of dict.

        max_iter : int, default: 10
            Maximal number of iterations KMeans will do for computing
            statistic.

        seed : int, default: 0
            Random seed for generating uniform data sets.

        trials : int, default: 10
            Number of data sets drawn as a reference.

        correction : bool, default: True
            If True, the correction is applied and the first feasible solution
            is selected. Otherwise the globally maximal GAP is used.

        Default: {'max_iter': 10, 'seed': 0, 'trials': 10, 'correction': True}

    verbose : bool, default: False
        If True, shows progress with tqdm.

    Attributes
    ----------

    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.

    labels_ :
        Labels of each point.

    estimators_ : List[KMeans]
        KMeans instances for n_clusters in range [min_clusters, max_clusters].

    scores_ : array, [max_clusters - min_clusters + 1, ?]
        Array with scores for each estimator in each row.

    n_clusters_ : int
        Estimated optimal number of clusters.

    best_score_ : float
        Score of the optimal estimator.

    best_ : KMeans
        The optimal estimator.

    """
    # TODO: Add example of usage.
    def __init__(self, max_clusters: int, min_clusters: int = 1,
                 n_jobs: int = 1, method: str = 'dunn',
                 distance: str = 'euclidean', init: str = 'percentile',
                 percentile: float = 95., max_iter: int = 100,
                 normalize_rows: bool = False, gap=None,
                 verbose: bool = False):
        super().__init__()
        assert method in {'dunn', 'gap'}
        assert min_clusters <= max_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_jobs = n_jobs
        self.method = method
        self.distance = distance
        self.init = init
        self.percentile = percentile
        self.max_iter = max_iter
        self.normalize_rows = normalize_rows
        self.gap = gap
        self.verbose = verbose

    def fit(self, X, y=None, pool: Pool=None):
        """Compute k-means clustering and estimate optimal number of clusters.

        Parameters
        ----------

        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.

        y : Ignored
            not used, present here for API consistency by convention.

        pool: Pool
            used for parallelization of computations reusing single pool
        """
        fit_kmeans = partial(_fit_kmeans, data=X, distance=self.distance,
                             init=self.init, percentile=self.percentile,
                             max_iter=self.max_iter,
                             normalize_rows=self.normalize_rows)
        n_clusters = range(self.min_clusters, self.max_clusters + 1)
        if self.verbose:
            n_clusters = tqdm.tqdm(n_clusters, leave=False, file=sys.stdout)

        method = make_picker(self.method, self.gap)

        processes = get_n_jobs(self.n_jobs)
        if processes == 1 or pool is None:
            self.estimators_ = [fit_kmeans(n_clusters=k) for k in n_clusters]
            self.scores_ = method.score(X, self.estimators_)
        elif pool is not None:
            self.estimators_ = pool.map(fit_kmeans, n_clusters)
            self.scores_ = method.score(X, self.estimators_, pool)
        else:
            with Pool(processes) as pool:
                self.estimators_ = pool.map(fit_kmeans, n_clusters)
                self.scores_ = method.score(X, self.estimators_, pool)

        best = method.select(self.scores_)

        if best is None:
            self.fitted_ = False
            self.n_clusters_ = None
            self.best_score_ = None
            self.best_ = None
            self.labels_ = None
            self.cluster_centers_ = None
        else:
            self.fitted_ = True
            self.n_clusters_ = best + self.min_clusters
            self.best_score_ = self.scores_[best]
            self.best_ = self.estimators_[best]
            self.labels_ = self.best_.labels_
            self.cluster_centers_ = self.best_.cluster_centers_

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
        check_is_fitted(self, 'best_')
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
        check_is_fitted(self, 'best_')
        return self.best_.transform(X)
