import numpy as np
from sklearn.base import BaseEstimator
from statsmodels.stats.stattools import medcouple

from ._stat_selector_mixin import StatSelectorMixin


def huberta_outliers(v):
    """
    M. Huberta, E.Vandervierenb (2008) An adjusted boxplot for skewed
    distributions, Computational Statistics and Data Analysis 52 (2008)
    5186–5201

    Parameters
    ----------
    v: array-like
        An array to filter outlier from.

    Returns
    -------
    Binary vector indicating all the outliers.
    """
    q1, q3 = np.quantile(v, q=[.25, .75])
    iqr = q3 - q1
    MC = medcouple(v)
    if MC >= 0:
        lower_bound = q1 - 1.5 * np.exp(-4 * MC) * iqr
        upper_bound = q3 + 1.5 * np.exp(3 * MC) * iqr
    else:
        lower_bound = q1 - 1.5 * np.exp(-3 * MC) * iqr
        upper_bound = q3 + 1.5 * np.exp(4 * MC) * iqr
    return np.logical_or(v < lower_bound, v > upper_bound)


# noinspection PyAttributeOutsideInit
class OutlierSelector(BaseEstimator, StatSelectorMixin):
    """Feature selector that removes outlier features w.r.t. mean or variance

    Huberta's outlier detection is applied to the features' characteristics
    and the outlying features are removed.

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

    keep_outliers: bool, optional, default: False
        When True, keeps outliers instead of inlier features.

    Attributes
    ----------
    vals_: array, shape (n_features,)
        Computed characteristic of each feature.

    selected_: array, shape (n_features,)
        Vector of binary selections of the informative features.
    """
    def __init__(self, stat: str, use_log: bool = False,
                 keep_outliers: bool = False):
        self.stat = stat
        self.use_log = use_log
        self.keep_outliers = keep_outliers

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
        outliers = huberta_outliers(self.vals_)
        if self.keep_outliers:
            self.selected_ = outliers
        else:
            self.selected_ = outliers == False
        return self
