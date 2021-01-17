import numpy as np
from sklearn.base import BaseEstimator

from divik.core import configurable

from ._stat_selector_mixin import StatSelectorMixin


def _medcouple_1d(y):
    """
    Calculates the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.

    Returns
    -------
    mc : float
        The medcouple statistic

    Notes
    -----
    The current algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [*] M. Huberta and E. Vandervierenb, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """

    # Parameter changes the algorithm to the slower for large n

    y = np.squeeze(np.asarray(y))
    if y.ndim != 1:
        raise ValueError("y must be squeezable to a 1-d array")

    y = np.sort(y)

    n = y.shape[0]
    if n % 2 == 0:
        mf = (y[n // 2 - 1] + y[n // 2]) / 2
    else:
        mf = y[(n - 1) // 2]

    z = y - mf
    lower = z[z <= 0.0]
    upper = z[z >= 0.0]
    upper = upper[:, None]
    standardization = upper - lower
    is_zero = np.logical_and(lower == 0.0, upper == 0.0)
    standardization[is_zero] = np.inf
    spread = upper + lower
    h = spread / standardization
    # GH5395
    num_ties = np.sum(lower == 0.0)
    if num_ties:
        # Replacements has -1 above the anti-diagonal, 0 on the anti-diagonal,
        # and 1 below the anti-diagonal
        replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
        replacements -= 2 * np.triu(replacements)
        # Convert diagonal to anti-diagonal
        replacements = np.fliplr(replacements)
        # Always replace upper right block
        h[:num_ties, -num_ties:] = replacements

    return np.median(h)


def medcouple(y, axis=0):
    """
    Calculate the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : {int, None}
        Axis along which the medcouple statistic is computed.  If `None`, the
        entire array is used.

    Returns
    -------
    mc : ndarray
        The medcouple statistic with the same shape as `y`, with the specified
        axis removed.

    Notes
    -----
    The current algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [*] M. Huberta and E. Vandervierenb, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """
    y = np.asarray(y, dtype=np.double)  # GH 4243
    if axis is None:
        return _medcouple_1d(y.ravel())

    return np.apply_along_axis(_medcouple_1d, axis, y)


def huberta_outliers(v):
    """Outlier detection method based on medcouple statistic.

    References
    ----------

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
    q1, q3 = np.quantile(v, q=[0.25, 0.75])
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
@configurable
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

    def __init__(self, stat: str, use_log: bool = False, keep_outliers: bool = False):
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
