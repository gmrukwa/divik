import logging
from abc import ABCMeta

import numpy as np
from sklearn.base import BaseEstimator

try:
    from sklearn.feature_selection._base import SelectorMixin
except ModuleNotFoundError:
    from sklearn.feature_selection.base import SelectorMixin

from divik.core import configurable


class StatSelectorMixin(SelectorMixin, metaclass=ABCMeta):
    """
    Transformer mixin that performs feature selection given a support mask

    This mixin provides a feature selector implementation with ``transform`` and
    ``inverse_transform`` functionality given that ``selected_`` is specified
    during ``fit``.

    Additionally, provides a ``_to_characteristics`` and ``_to_raw`` implementations
    given ``stat``, optionally ``use_log`` and ``preserve_high``.
    """

    def _to_characteristics(self, X):
        """Extract & normalize characteristics from data"""
        if self.stat == "mean":
            vals = np.mean(X, axis=0)
        elif self.stat == "var":
            vals = np.var(X, axis=0)
        elif self.stat == "cv":
            vals = np.std(X, axis=0) / np.mean(X, axis=0)
        elif callable(self.stat):
            vals = self.stat(X)
            if vals.size != X.shape[1]:
                raise RuntimeError(
                    "Computed statistic shape mismatch {0}".format(vals.shape)
                )
        else:
            msg = 'stat must be one of {"cv", "mean", "var"} or callable'
            logging.error(msg)
            raise ValueError(msg)

        if hasattr(self, "use_log") and self.use_log:
            if np.any(vals < 0):
                logging.error(
                    "Feature characteristic cannot be negative with log filtering"
                )
                raise ValueError(
                    "Feature characteristic cannot be negative with log filtering"
                )
            vals = np.log(vals)

        if hasattr(self, "preserve_high") and not self.preserve_high:
            vals = -vals

        return vals

    def _to_raw(self, threshold):
        """Convert threshold to the feature characteristic space"""
        if hasattr(self, "preserve_high") and not self.preserve_high:
            threshold = -threshold
        if hasattr(self, "use_log") and self.use_log:
            threshold = np.exp(threshold)
        return threshold

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


@configurable
class NoSelector(BaseEstimator, StatSelectorMixin):
    """Dummy selector to use when no selection is supposed to be made."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Pass data forward

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors to pass.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self
        """
        self.selected_ = np.ones((X.shape[1],), dtype=bool)
        return self
