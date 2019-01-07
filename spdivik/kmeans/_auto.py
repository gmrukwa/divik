from functools import partial
from multiprocessing import Pool
import os

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from spdivik.kmeans._core import KMeans
from spdivik.score import DunnPicker, GapPicker


def _fit_kmeans(n_clusters: int, data: np.ndarray, distance: str = 'euclidean',
                init: str = 'percentile', percentile: float = 95.,
                max_iter: int = 100, normalize_rows: bool = False):
    kmeans = KMeans(n_clusters=n_clusters, distance=distance, init=init,
                    percentile=percentile, max_iter=max_iter,
                    normalize_rows=normalize_rows)
    kmeans.fit(data)
    return kmeans


def _processes(n_jobs: int) -> int:
    cpus = os.cpu_count() or 1
    processes = ((n_jobs + int(n_jobs <= 0)) % cpus) or cpus
    return processes


class AutoKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, min_clusters: int, max_clusters: int, n_jobs: int = 1,
                 method: str = 'dunn', distance: str = 'euclidean',
                 init: str = 'percentile', percentile: float = 95.,
                 max_iter: int = 100, normalize_rows: bool = False,
                 gap_max_iter: int = 10, gap_seed: int = 0,
                 gap_trials: int = 10, verbose: bool=False):
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
        self.gap_max_iter = gap_max_iter
        self.gap_seed = gap_seed
        self.gap_trials = gap_trials
        self.verbose = verbose

    def fit(self, X, y=None):
        fit_kmeans = partial(_fit_kmeans, data=X, distance=self.distance,
                             init=self.init, percentile=self.percentile,
                             max_iter=self.max_iter,
                             normalize_rows=self.normalize_rows)
        n_clusters = range(self.min_clusters, self.max_clusters + 1)
        if self.verbose:
            n_clusters = tqdm.tqdm(n_clusters)

        if self.method == 'dunn':
            method = DunnPicker()
        elif self.method == 'gap':
            method = GapPicker(self.gap_max_iter, self.gap_seed, self.gap_trials)
        else:
            raise ValueError('Unknown quality measure {0}'.format(self.method))

        processes = _processes(self.n_jobs)
        if processes == 1:
            self.estimators_ = [fit_kmeans(n_clusters=k) for k in n_clusters]
            self.scores_ = method.score(X, self.estimators_)
        else:
            with Pool(processes) as pool:
                self.estimators_ = pool.map(fit_kmeans, n_clusters)
                self.scores_ = method.score(X, self.estimators_, pool)

        best = method.select(self.scores_)

        self.n_clusters_ = best + self.min_clusters
        self.best_score_ = self.scores_[best]
        self.best_ = self.estimators_[best]
        self.labels_ = self.best_.labels_
        self.cluster_centers_ = self.best_.cluster_centers_

        return self

    def predict(self, X):
        check_is_fitted(self, 'best_')
        return self.best_.predict(X)

    def transform(self, X):
        check_is_fitted(self, 'best_')
        return self.best_.transform(X)
