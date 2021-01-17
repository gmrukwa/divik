"""Implementation of PCA with data-driven variance explanation limit"""
import warnings

import numpy as np
from kneed import KneeLocator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from divik.core import configurable


def knee(explained_variance) -> int:
    """Find empirical knee point for explained variance"""
    xaxis = np.arange(explained_variance.size, dtype=int)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            limit = KneeLocator(
                x=xaxis,
                y=explained_variance,
                S=1.0,
                direction="increasing",
                curve="concave",
            ).knee
    except IndexError:  # This is needed for kneed >= 0.5.3
        limit = None
    if limit is not None:
        return limit
    return explained_variance.size


@configurable
class KneePCA(BaseEstimator, TransformerMixin):
    """Principal component analysis (PCA) with knee method

    PCA with automated components selection based on knee method
    over cumulative explained variance. Remaining components are
    discarded.

    Parameters
    -----------
    whiten : bool, optional (default False)
        When True (False by default) the ``pca_.components_`` vectors are
        multiplied by the square root of n_samples and then divided by the
        singular values to ensure uncorrelated outputs with unit
        component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    refit : bool, optional (default False)
        When ``True`` (``False`` by default) the ``pca_`` is re-fit with the smaller
        number of components. This could reduce memory footprint, but
        requires training fitting PCA.

    Attributes
    ----------
    pca_ : PCA
        Fit PCA estimator.

    n_components_ : int
        The number of selected components.
    """

    def __init__(self, whiten: bool = False, refit: bool = False):
        self.whiten = whiten
        self.refit = refit

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Y: Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Note: random_state is not used in this config!
        self.pca_ = PCA(
            n_components=None,
            copy=True,
            whiten=self.whiten,
            svd_solver="full",
            tol=0.0,
            iterated_power="auto",
            random_state=None,
        ).fit(X)
        self.n_components_ = knee(np.cumsum(self.pca_.explained_variance_ratio_))
        if self.refit:
            self.pca_ = PCA(
                n_components=self.n_components_,
                copy=True,
                whiten=self.whiten,
                svd_solver="full",
                tol=0.0,
                iterated_power="auto",
                random_state=None,
            ).fit(X)
        return self

    def transform(self, X, y=None):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        Examples
        --------

        >>> import numpy as np
        >>> from divik.feature_extraction import KneePCA
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> pca = KneePCA(refit=True)
        >>> pca.fit(X)
        KneePCA(refit=True)
        >>> pca.transform(X) # doctest: +SKIP
        """
        loads = self.pca_.transform(X)
        return loads[:, : self.n_components_]

    def inverse_transform(self, X):
        """Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where ``n_samples`` is the number of samples
            and ``n_components`` is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)

        Notes
        -----
        If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        n_missing = self.pca_.n_components_ - self.n_components_
        missing = np.zeros((X.shape[0], n_missing))
        full = np.hstack([X, missing])
        return self.pca_.inverse_transform(full)
