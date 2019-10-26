"""Methods of data-driven feature selection.

_feature_selection.py

Copyright 2019 Grzegorz Mrukwa

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin

import divik._matlab_legacy as ml


class GMMSelector(BaseEstimator, SelectorMixin):
    """Feature selector that removes low- or high- mean or variance features

    Gaussian Mixture Modeling is applied to the features' characteristics
    and components are obtained. Crossing points of the components are
    considered candidate thresholds. Out of these up to ``n_candidates``
    components are removed in such a way that at least ``min_features`` or
    ``min_features_rate`` features are retained.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    stat: {'mean', 'var'}
        Kind of statistic to be computed out of the feature.

    use_log: bool, optional, default: False
        Whether to use the logarithm of feature characteristic instead of the
        characteristic itself. This may improve feature filtering performance,
        depending on the distribution of features, however all the
        characteristics (mean, variance) have to be positive for that -
        filtering will fail otherwise. This is useful for specific cases in
        biology where the distribution of data may actually require this option
        for any efficient filtering.

    n_candidates: int, optional, default: None
        How many candidate thresholds to use at most. ``0`` preserves all the
        features (all candidate thresholds are discarded), ``None`` allows to
        remove all but one component (all candidate thresholds are retained).
        Negative value means to discard up to all but ``-n_candidates``
        candidates, e.g. ``-1`` will retain at least two components (one
        candidate threshold is removed).

    min_features: int, optional, default: 1
        How many features must be preserved. Candidate thresholds are tested
        against this value, and if they retain less features, less conservative
        thresholds is selected.

    min_features_rate: float, optional, default: 0.0
        Similar to ``min_features`` but relative to the input data features
        number.

    preserve_high: bool, optional, default: True
        Whether to preserve the high-characteristic features or
        low-characteristic ones.

    max_components: int, optional, default: 10
        The maximum number of components used in the GMM decomposition.

    Attributes
    ----------
    vals_: array, shape (n_features,)
        Computed characteristic of each feature.

    threshold_: float
        Threshold value to filter the features by the characteristic.

    raw_threshold_: float
        Threshold value mapped back to characteristic space (no logarithm, etc.)

    selected_: array, shape (n_features,)
        Vector of binary selections of the informative features.

    Examples
    --------
    >>> import numpy as np
    >>> import divik.feature_selection as fs
    >>> np.random.seed(42)
    >>> labels = np.concatenate([30 * [0] + 20 * [1] + 30 * [2] + 40 * [3]])
    >>> data = labels * 5 + np.random.randn(*labels.shape)
    >>> fs.GMMSelector('mean').fit_transform(data)
    array([[14.78032811 15.35711257 ... 15.75193303]])
    >>> fs.GMMSelector('mean', preserve_high=False).fit_transform(data)
    array([[ 0.49671415 -0.1382643  ... -0.29169375]])
    >>> fs.GMMSelector('mean', n_discard=-1).fit_transform(data)
    array([[10.32408397  9.61491772 ... 15.75193303]])


    """
    # TODO: Improve docstring so it will work with doctest
    def __init__(self, stat: str, use_log: bool = False,
                 n_candidates: int = None, min_features: int = 1,
                 min_features_rate: float = .0, preserve_high: bool = True,
                 max_components: int = 10):
        if stat not in {'mean', 'var'}:
            raise ValueError('stat must be one of {"mean", "var"}')
        self.stat = stat
        self.use_log = use_log
        self.n_candidates = n_candidates
        self.min_features = min_features
        self.min_features_rate = min_features_rate
        self.preserve_high = preserve_high
        self.max_components = max_components

    def _to_characteristics(self, X):
        """Extract & normalize characteristics from data"""
        if self.stat == 'mean':
            vals = np.mean(X, axis=0)
        elif self.stat == 'var':
            vals = np.var(X, axis=0)
        else:
            raise ValueError('stat must be one of {"mean", "var"}')

        if self.use_log:
            if np.any(vals < 0):
                raise ValueError("Feature characteristic cannot be negative "
                                 "with log filtering")
            vals = np.log(vals)

        if not self.preserve_high:
            vals = -vals
        
        return vals

    def _to_raw(self, threshold):
        """Convert threshold to the feature characteristic space"""
        if not self.preserve_high:
            threshold = -threshold
        if self.use_log:
            threshold = np.exp(threshold)
        return threshold

    def fit(self, X, y=None):
        """Learn data-driven feature thresholds from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute feature characteristic.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self
        """
        self.vals_ = self._to_characteristics(X)
        thrs = ml.find_thresholds(  # the translation is due to MATLAB's problem
            self.vals_ - self.vals_.min(),
            max_components=self.max_components,
            throw_on_engine_error=False) + self.vals_.min()
        n_candidates = len(thrs) if self.n_candidates is None \
            else self.n_candidates
        desired_thrs = thrs[:n_candidates]
        min_features = max(
            self.min_features, self.min_features_rate * X.shape[1])
        for thr in reversed(desired_thrs):
            selected = self.vals_ >= thr
            if selected.sum() >= min_features:
                break
        else:
            selected = np.ones((X.shape[1],), dtype=bool)
            thr = -np.inf
        self.threshold_ = thr
        self.raw_threshold_ = self._to_raw(thr)
        self.selected_ = selected
        return self

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected

        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """
        return self.selected_


class HighAbundanceAndVarianceSelector(BaseEstimator, SelectorMixin):
    """Feature selector that removes low-mean and low-variance features

    Exercises ``GMMSelector`` to filter out the low-abundance noise features
    and select high-variance informative features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Parameters
    ----------
    use_log: bool, optional, default: False
        Whether to use the logarithm of feature characteristic instead of the
        characteristic itself. This may improve feature filtering performance,
        depending on the distribution of features, however all the
        characteristics (mean, variance) have to be positive for that -
        filtering will fail otherwise. This is useful for specific cases in
        biology where the distribution of data may actually require this option
        for any efficient filtering.

    min_features: int, optional, default: 1
        How many features must be preserved.

    min_features_rate: float, optional, default: 0.0
        Similar to ``min_features`` but relative to the input data features
        number.

    max_components: int, optional, default: 10
        The maximum number of components used in the GMM decomposition.

    Attributes
    ----------
    abundance_selector_: GMMSelector
        Selector used to filter out the noise component.

    variance_selector_: GMMSelector
        Selector used to filter out the non-informative features.

    selected_: array, shape (n_features,)
        Vector of binary selections of the informative features.

    Examples
    --------
    >>> import numpy as np
    >>> import divik.feature_selection as fs
    >>> np.random.seed(42)
    >>> # Data in this case must be carefully crafted
    >>> labels = np.concatenate([30 * [0] + 20 * [1] + 30 * [2] + 40 * [3]])
    >>> data = np.vstack(100 * [labels * 10.])
    >>> data += np.random.randn(*data.shape)
    >>> sub = data[:, :-40]
    >>> sub += 5 * np.random.randn(*sub.shape)
    >>> # Label 0 has low abundance but high variance
    >>> # Label 3 has low variance but high abundance
    >>> # Label 1 and 2 has not-lowest abundance and high variance
    >>> selector = fs.HighAbundanceAndVarianceSelector().fit(data)
    >>> selector.transform(labels.reshape(1,-1))
    array([[1 1 1 1 1 ...2 2 2]])

    """
    # TODO: Improve docstring so it will work with doctest
    def __init__(self, use_log: bool = False, min_features: int = 1,
                 min_features_rate: float = 0., max_components: int = 10):
        self.use_log = use_log
        self.min_features = min_features
        self.min_features_rate = min_features_rate
        self.max_components = max_components

    def fit(self, X, y=None):
        """Learn data-driven feature thresholds from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute feature characteristic.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self
        """
        min_features = max(
            self.min_features, self.min_features_rate * X.shape[1])

        self.abundance_selector_ = GMMSelector(
            'mean', use_log=self.use_log, n_candidates=1,
            min_features=min_features, preserve_high=True,
            max_components=self.max_components
        ).fit(X)
        filtered = self.abundance_selector_.transform(X)
        self.selected_ = self.abundance_selector_.selected_.copy()

        self.variance_selector_ = GMMSelector(
            'var', use_log=self.use_log, n_candidates=None,
            min_features=min_features, preserve_high=True,
            max_components=self.max_components
        ).fit(filtered)
        self.selected_[self.selected_] = self.variance_selector_.selected_

        return self

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected

        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """
        return self.selected_
