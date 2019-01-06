from functools import partial
from multiprocessing import Pool

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

from spdivik.kmeans._core import KMeans
from spdivik.score._dunn import dunn


def _fit_kmeans(data: np.ndarray, n_clusters: int, distance: str = 'euclidean',
                init: str = 'percentile', percentile: float = 95.,
                max_iter: int = 100, normalize_rows: bool = False):
    kmeans = KMeans(n_clusters=n_clusters, distance=distance, init=init,
                    percentile=percentile, max_iter=max_iter,
                    normalize_rows=normalize_rows)
    kmeans.fit(data)
    return kmeans


class AutoKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, min_clusters: int, max_clusters: int, n_jobs: int = 1,
                 method: str = 'dunn', distance: str = 'euclidean',
                 init: str = 'percentile', percentile: float = 95.,
                 max_iter: int = 100, normalize_rows: bool = False):
        super().__init__()
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

    def fit(self, X, y=None):
        fit_kmeans = partial(_fit_kmeans, data=X, distance=self.distance,
                             init=self.init, percentile=self.percentile,
                             max_iter=self.max_iter,
                             normalize_rows=self.normalize_rows)
        n_clusters = range(self.min_clusters, self.max_clusters + 1)

        # TODO: Generalize for GAP
        score = partial(dunn, data=X)

        if self.n_jobs == 1:
            self.estimators_ = [fit_kmeans(n_clusters=k) for k in n_clusters]
            self.scores_ = [score(estimator) for estimator in self.estimators_]
        else:
            with Pool(self.n_jobs) as pool:
                self.estimators_ = pool.map(fit_kmeans, n_clusters)
                self.scores_ = pool.map(score, self.estimators_)

        # TODO: Generalize for other scenarios
        best = int(np.argmax(self.scores_))

        self.n_clusters_ = best + self.min_clusters
        self.best_score_ = self.scores_[best]
        self.best_ = self.estimators_[best]
        self.labels_ = self.best_.labels_
        self.cluster_centers_ = self.best_.cluster_centers_

        return self

    def predict(self, X):
        return self.best_.predict(X)

    def transform(self, X):
        return self.best_.transform(X)
