from sklearn.base import BaseEstimator

from divik.core import configurable
from divik.feature_selection._stat_selector_mixin import SelectorMixin

from ._exims import exims
from ._selection import select_features


@configurable
class EximsSelector(BaseEstimator, SelectorMixin):
    """Select features based on their spatial distribution

    Preserves features that yield biologically plausible structures.

    References
    ----------

    Wijetunge, Chalini D., et al. "EXIMS: an improved data analysis
    pipeline based on a new peak picking method for EXploring Imaging
    Mass Spectrometry data." Bioinformatics 31.19 (2015): 3198-3206.
    https://academic.oup.com/bioinformatics/article/31/19/3198/212150
    """

    def __init__(self):
        super(EximsSelector, self).__init__()

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

    def fit(self, X, y=None, xy=None):
        """Learn data-driven feature thresholds from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute feature characteristic.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        xy : array-like, shape (n_samples, 2)
            Spatial coordinates of the samples. Expects integers,
            indices over am image.

        Returns
        -------
        self
        """
        if xy is None:
            raise ValueError("xy coordinates are required")
        self.structness_ = exims(X, *xy.T)
        features_selection = select_features(self.structness_)
        self.threshold_ = features_selection.threshold
        self.selected_ = features_selection.selection
        return self
