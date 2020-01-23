from abc import ABCMeta, abstractmethod
from functools import partial
import sys
from typing import Tuple, Union

import numpy as np
import pandas as pd
import tqdm
from scipy.spatial import distance as dist
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from divik import _summary as summary, feature_selection as fs
from divik.core import context_if, DivikResult, normalize_rows, maybe_pool


class DiviKBase(BaseEstimator, ClusterMixin, TransformerMixin, metaclass=ABCMeta):
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
                 k_max: int = 10,
                 sample_size: int = 10000,
                 normalize_rows: bool = None,
                 use_logfilters: bool = False,
                 filter_type='gmm',
                 n_jobs: int = None,
                 random_seed: int = 0,  # TODO: Rework to use RandomState
                 verbose: bool = False):
        self.gap_trials = gap_trials
        self.leaf_size = leaf_size
        self.max_iter = max_iter
        self.distance = distance
        self.minimal_size = minimal_size
        self.rejection_size = rejection_size
        self.rejection_percentage = rejection_percentage
        self.minimal_features_percentage = minimal_features_percentage
        self.features_percentage = features_percentage
        self.k_max = k_max
        self.sample_size = sample_size
        self.normalize_rows = normalize_rows
        self.use_logfilters = use_logfilters
        self.filter_type = filter_type
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.verbose = verbose
        self._validate_arguments()

    def _validate_feature_selection(self):
        if self.minimal_features_percentage < 0 \
                or self.minimal_features_percentage > 1:
            raise ValueError('minimal_features_percentage must be in range'
                             ' [0, 1]')
        if self.features_percentage < 0 or self.features_percentage > 1:
            raise ValueError('features_percentage must be in range [0, 1]')
        if self.features_percentage < self.minimal_features_percentage:
            raise ValueError('features_percentage must be higher than or equal'
                             ' to minimal_features_percentage')
        if self.filter_type not in ['gmm', 'outlier', 'auto', 'none']:
            raise ValueError(
                "filter_type must be in ['gmm', 'outlier', 'auto', 'none']")

    def _validate_clustering(self):
        if self.gap_trials <= 0:
            raise ValueError('gap_trials must be greater than 0')
        if self.leaf_size < 0:
            raise ValueError('leaf_size must be greater than 0')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be greater than 0')
        if self.minimal_size is not None and self.minimal_size < 0:
            raise ValueError('minimal_size must be greater or equal to 0')

    def _validate_arguments(self):
        self._validate_clustering()
        self._validate_feature_selection()

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

        with context_if(self.verbose, tqdm.tqdm,
                total=X.shape[0], file=sys.stdout) as progress:
            self.result_ = self._divik(X, progress)

        if self.result_ is None:
            self.labels_ = np.zeros((X.shape[0],), dtype=int)
            self.paths_ = {0: (0,)}
        else:
            self.labels_, self.paths_ = summary.merged_partition(
                self.result_, return_paths=True)

        self.reverse_paths_ = {
            value: key for key, value in self.paths_.items()}

        if self.result_ is None:
            self.filters_ = np.ones([1, X.shape[1]], dtype=bool)
        else:
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
            return self.distance == 'correlation'
        return self.normalize_rows

    def _gmm_filter(self):
        return fs.HighAbundanceAndVarianceSelector(
            use_log=self.use_logfilters,
            min_features_rate=self.minimal_features_percentage)

    def _outlier_filter(self):
        return fs.OutlierAbundanceAndVarianceSelector(
            use_log=self.use_logfilters,
            min_features_rate=self.minimal_features_percentage,
            p=self.features_percentage)

    def _feature_selector(self, n_features):
        if (self.filter_type == 'auto' and n_features > 250) \
                or self.filter_type == 'gmm':
            return self._gmm_filter()
        if self.filter_type == 'auto' or self.filter_type == 'outlier':
            return self._outlier_filter()
        if self.filter_type == 'none':
            return fs.NoSelector()
        raise ValueError("Unknown filter type: %s" % self.filter_type)

    @abstractmethod
    def _divik(self, X, progress):
        pass

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
        distances = np.hstack([
            dist.cdist(
                X[:, selector], centroid[np.newaxis, selector], self.distance)
            for selector, centroid in zip(self.filters_, self.centroids_)
        ])
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
