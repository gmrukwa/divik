import numpy as np
from sklearn.base import BaseEstimator

from divik.core import configurable

from ._stat_selector_mixin import StatSelectorMixin


# noinspection PyAttributeOutsideInit
@configurable
class PercentageSelector(BaseEstimator, StatSelectorMixin):
    """Feature selector that removes / preserves top some percent of features

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

    keep_top: bool, optional, default: True
        When True, keeps features with highest value of the characteristic.

    p: float, optional, default: 0.2
        Rate of features to keep.

    Attributes
    ----------
    vals_: array, shape (n_features,)
        Computed characteristic of each feature.

    threshold_: float
        Value of the threshold used for filtering

    selected_: array, shape (n_features,)
        Vector of binary selections of the informative features.
    """

    def __init__(
        self, stat: str, use_log: bool = False, keep_top: bool = True, p: float = 0.2
    ):
        self.stat = stat
        self.use_log = use_log
        self.keep_top = keep_top
        self.p = p

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
        if self.keep_top:
            self.threshold_ = np.quantile(self.vals_, q=1 - self.p)
            self.selected_ = self.threshold_ <= self.vals_
        else:
            self.threshold_ = np.quantile(self.vals_, q=self.p)
            self.selected_ = self.threshold_ >= self.vals_
        return self
