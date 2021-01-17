import numpy as np
from sklearn.base import BaseEstimator

from divik.core import build, configurable

from ._gmm_selector import GMMSelector
from ._outlier import OutlierSelector
from ._percentage_selector import PercentageSelector
from ._stat_selector_mixin import NoSelector, SelectorMixin


@configurable
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

    def __init__(
        self,
        use_log: bool = False,
        min_features: int = 1,
        min_features_rate: float = 0.0,
        max_components: int = 10,
    ):
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
        min_features = max(self.min_features, self.min_features_rate * X.shape[1])

        if min_features == X.shape[1]:
            self.selected_ = np.ones((X.shape[1],), dtype=bool)
            return self

        self.abundance_selector_ = GMMSelector(
            "mean",
            use_log=self.use_log,
            n_candidates=1,
            min_features=min_features,
            preserve_high=True,
            max_components=self.max_components,
        ).fit(X)
        filtered = self.abundance_selector_.transform(X)
        self.selected_ = self.abundance_selector_.selected_.copy()

        self.variance_selector_ = GMMSelector(
            "var",
            use_log=self.use_log,
            n_candidates=None,
            min_features=min_features,
            preserve_high=True,
            max_components=self.max_components,
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


EPS = 10e-6


# noinspection PyAttributeOutsideInit
@configurable
class OutlierAbundanceAndVarianceSelector(BaseEstimator, SelectorMixin):
    def __init__(
        self, use_log: bool = False, min_features_rate: float = 0.01, p: float = 0.2
    ):
        self.use_log = use_log
        self.min_features_rate = min_features_rate
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
        self.abundance_selector_, a_selected = self._fit_abundance(X)
        filtered = X[:, a_selected]
        self.variance_selector_, v_selected = self._fit_variance(filtered, a_selected)
        self.selected_ = a_selected
        self.selected_[a_selected] = v_selected
        return self

    def _fit_abundance(self, X):
        selector = OutlierSelector(
            stat="mean", use_log=self.use_log, keep_outliers=False
        ).fit(X)
        selected = selector.selected_
        inlier = selector.vals_[selected][0]
        over_inlier = selector.vals_ > inlier
        selected[over_inlier] = True
        p = selected.mean()
        if p < self.min_features_rate or p >= 1 - EPS:
            selector = PercentageSelector(
                stat="mean", use_log=self.use_log, keep_top=True, p=1.0 - self.p
            ).fit(X)
            selected = selector.selected_
        return selector, selected

    def _fit_variance(self, X, old_selected):
        corrected_min = self.min_features_rate / old_selected.mean()
        corrected_p = self.p / old_selected.mean()

        selector = OutlierSelector(
            stat="var", use_log=self.use_log, keep_outliers=True
        ).fit(X)
        selected = selector.selected_
        inlier = selector.vals_[selected == 0][0]
        under_inlier = selector.vals_ < inlier
        selected[under_inlier] = False
        p = selected.mean()

        if p < corrected_min or p >= 1 - EPS:
            selector = PercentageSelector(
                stat="var", use_log=self.use_log, keep_top=True, p=corrected_p
            ).fit(X)
            selected = selector.selected_
        return selector, selected

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


def make_specialized_selector(name, n_features, **kwargs):
    """Create a selector by name (``gmm``, ``outlier``, ``none`` or ``auto``)

    ``auto`` switches to ``gmm`` if there is more than 250 features, ``outlier`` below.
    """
    if name == "auto":
        name = "gmm" if n_features > 250 else "outlier"
    filter_cls = {
        "gmm": HighAbundanceAndVarianceSelector,
        "outlier": OutlierAbundanceAndVarianceSelector,
        "none": NoSelector,
    }[name]
    return build(filter_cls, **kwargs)
