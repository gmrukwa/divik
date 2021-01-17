import logging

import numpy as np
from sklearn.base import BaseEstimator

import divik._matlab_legacy as ml
from divik.core import configurable

from ._stat_selector_mixin import StatSelectorMixin


@configurable
class GMMSelector(BaseEstimator, StatSelectorMixin):
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

    def __init__(
        self,
        stat: str,
        use_log: bool = False,
        n_candidates: int = None,
        min_features: int = 1,
        min_features_rate: float = 0.0,
        preserve_high: bool = True,
        max_components: int = 10,
    ):
        if stat not in {"cv", "mean", "var"} and not callable(stat):
            msg = 'stat must be one of {"cv", "mean", "var"} or callable'
            logging.error(msg)
            raise ValueError(msg)
        self.stat = stat
        self.use_log = use_log
        self.n_candidates = n_candidates
        self.min_features = min_features
        self.min_features_rate = min_features_rate
        self.preserve_high = preserve_high
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
        self.vals_ = self._to_characteristics(X)
        thrs = ml.find_thresholds(self.vals_, max_components=self.max_components)
        n_candidates = len(thrs) if self.n_candidates is None else self.n_candidates
        desired_thrs = thrs[:n_candidates]
        min_features = max(self.min_features, self.min_features_rate * X.shape[1])
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
