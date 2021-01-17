import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from divik.core import configurable
from divik.core import seed as seed_
from divik.feature_extraction import KneePCA

from ._core import BaseSampler


@configurable
class UniformSampler(BaseSampler):
    """Samples uniformly from the boundaries of the data

    Parameters
    -----------
    n_rows : int, optional (default None)
        Allows to limit the number of rows in the drawn samples

    n_samples : int, optional (default None)
        Allows to limit the number of samples when iterating

    Attributes
    ----------
    shape_ : (n_rows, n_cols)
        Shape of the drawn samples

    scaler_ : MinMaxScaler
        Scaler ensuring the proper ranges
    """

    def __init__(self, n_rows: int = None, n_samples: int = None):
        self.n_rows = n_rows
        self.n_samples = n_samples

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

        Y: Ignored.

        Returns
        -------
        self : UniformSampler
            Returns the instance itself.
        """
        if self.n_rows is None:
            n_rows = X.shape[0]
        else:
            n_rows = self.n_rows
        self.shape_ = n_rows, X.shape[1]
        self.scaler_ = MinMaxScaler().fit(X)
        return self

    def get_sample(self, seed):
        """Return specific sample

        Parameters
        ----------
        seed : int
            The seed to use to draw the sample

        Returns
        -------
        sample : array_like, (*self.shape_)
            Returns the drawn sample
        """
        with seed_(seed):
            unscaled = np.random.random_sample(self.shape_)
        return self.scaler_.inverse_transform(unscaled)


@configurable
class UniformPCASampler(BaseSampler):
    """Rotation-invariant uniform sampling

    Parameters
    -----------
    n_rows : int, optional (default None)
        Allows to limit the number of rows in the drawn samples

    n_samples : int, optional (default None)
        Allows to limit the number of samples when iterating

    whiten : bool, optional (default False)
        When True (False by default) the `pca_.components_` vectors are
        multiplied by the square root of n_samples and then divided by the
        singular values to ensure uncorrelated outputs with unit
        component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    refit : bool, optional (default False)
        When True (False by default) the `pca_` is re-fit with the smaller
        number of components. This could reduce memory footprint, but
        requires training fitting PCA.

    pca: {'knee', 'full'}, default 'knee'
        Specifies whether to train full or knee PCA.

    Attributes
    ----------
    pca_ : KneePCA or PCA
        PCA transform which provided rotation-invariance

    sampler_ : UniformSampler
        Sampler from the transformed distribution
    """

    def __init__(
        self,
        n_rows: int = None,
        n_samples: int = None,
        whiten: bool = False,
        refit: bool = False,
        pca: str = "knee",
    ):
        self.n_rows = n_rows
        self.n_samples = n_samples
        self.whiten = whiten
        self.refit = refit
        self.pca = pca

    def _make_pca(self):
        if self.pca == "knee":
            return KneePCA(whiten=self.whiten, refit=self.refit)
        if self.pca == "full":
            # Note: random_state is not used in this config!
            return PCA(
                n_components=None,
                copy=True,
                whiten=self.whiten,
                svd_solver="full",
                tol=0.0,
                iterated_power="auto",
                random_state=None,
            )
        logging.error("Unsupported pca value: {}".format(self.pca))
        raise ValueError("Unsupported pca value: {}".format(self.pca))

    def fit(self, X, y=None):
        """Fit the model from data in X.

        PCA is fit to estimate the rotation and UniformSampler is
        fit to transformed data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            and n_features is the number of features.

        Y: Ignored.

        Returns
        -------
        self : UniformPCASampler
            Returns the instance itself.
        """
        self.pca_ = self._make_pca()
        transformed = self.pca_.fit_transform(X)
        self.sampler_ = UniformSampler(n_rows=self.n_rows).fit(transformed)
        return self

    def get_sample(self, seed):
        """Return specific sample

        Sample is generated from transformed distribution and transformed
        back to the original space.

        Parameters
        ----------
        seed : int
            The seed to use to draw the sample

        Returns
        -------
        sample : array_like, (*self.shape_)
            Returns the drawn sample
        """
        transformed = self.sampler_.get_sample(seed)
        return self.pca_.inverse_transform(transformed)
