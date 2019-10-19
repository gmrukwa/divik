from functools import reduce
from multiprocessing import Pool
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import tqdm

import divik.distance as dst
from divik.kmeans._core import normalize_rows
import divik.predefined as predefined
import divik.summary as summary


class DiviK(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self,
                 gap_trials: int = 10,
                 distance_percentile: float = 99.,
                 max_iter: int = 100,
                 distance: str = dst.KnownMetric.correlation.value,
                 minimal_size: int = None,
                 rejection_size: int = None,
                 rejection_percentage: float = None,
                 minimal_features_percentage: float = .01,
                 fast_kmeans_iters: int = 10,
                 k_max: int = 10,
                 normalize_rows: bool = None,
                 use_logfilters: bool = False,
                 n_jobs: int = None,
                 verbose: bool = False):
        if distance not in list(dst.KnownMetric):
            raise ValueError('Unknown distance: %s' % distance)

        self.gap_trials = gap_trials
        self.distance_percentile = distance_percentile
        self.max_iter = max_iter
        self.distance = distance
        self.minimal_size = minimal_size
        self.rejection_size = rejection_size
        self.rejection_percentage = rejection_percentage
        self.minimal_features_percentage = minimal_features_percentage
        self.fast_kmeans_iters = fast_kmeans_iters
        self.k_max = k_max
        self.normalize_rows = normalize_rows
        self.use_logfilters = use_logfilters
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        n_cpu = os.cpu_count()
        n_jobs = 1 if self.n_jobs is None else self.n_jobs
        n_jobs = (n_jobs + n_cpu) % n_cpu or n_cpu

        if self.normalize_rows is None:
            if self.distance == dst.KnownMetric.correlation.value:
                normalize_rows = True
            else:
                normalize_rows = False
        else:
            normalize_rows = self.normalize_rows

        minimal_size = int(X.shape[0] * 0.001) if self.minimal_size is None \
            else self.minimal_size

        with Pool(n_jobs) as pool,\
                tqdm.tqdm(total=X.shape[0], leave=self.verbose) as progress:
            divik = predefined.basic(
                gap_trials=self.gap_trials,
                distance_percentile=self.distance_percentile,
                iters_limit=self.max_iter,
                distance=self.distance,
                minimal_size=minimal_size,
                rejection_size=self.rejection_size,
                rejection_percentage=self.rejection_percentage,
                minimal_features_percentage=self.minimal_features_percentage,
                fast_kmeans_iters=self.fast_kmeans_iters,
                k_max=self.k_max,
                correction_of_gap=True,
                normalize_rows=normalize_rows,
                use_logfilters=self.use_logfilters,
                pool=pool,
                progress_reporter=progress if self.verbose else None
            )
            self.result_ = divik(X)

        self.labels_, self.paths_ = summary.merged_partition(self.result_,
                                                             return_paths=True)
        self.reverse_paths_ = {value: key for key, value in self.paths_.items()}
        self.centroids_ = pd.DataFrame(X).groupby(self.labels_).mean().values
        self.depth_ = summary.depth(self.result_)
        self.n_clusters_ = summary.total_number_of_clusters(self.result_)

        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def fit_transform(self, X, y=None, **fit_params):
        # TODO: optimize
        return self.fit(X).transform(X)

    def transform(self, X, with_path: bool = False):
        if self.normalize_rows is None:
            if self.distance == dst.KnownMetric.correlation.value:
                normalize_rows_ = True
            else:
                normalize_rows_ = False
        else:
            normalize_rows_ = self.normalize_rows

        if normalize_rows_:
            X = normalize_rows(X)

        distance = dst.ScipyDistance(dst.KnownMetric[self.distance])

        # TODO: optimize
        distances, paths = [], []
        for row in X:
            division = self.result_
            path = []
            while division is not None:
                selectors = division.filters
                restricted = reduce(np.logical_and, selectors.values(), True)
                local_X = row[np.newaxis, restricted]
                d = distance(local_X, division.centroids)
                assert d.shape[0] == 1 or d.shape[1] == 1
                d = d.ravel()
                label = np.argmin(d.ravel())
                path.append(label)
                division = division.subregions[label]
            path = tuple(path)
            distances.append(d[label])
            paths.append(path)

        if with_path:
            return np.array(distances), paths

        return np.array(distances)

    def predict(self, X):
        _, paths = self.transform(X, with_path=True)
        return np.array([self.reverse_paths_[path] for path in paths])
