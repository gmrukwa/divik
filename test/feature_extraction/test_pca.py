import unittest

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs

from divik.feature_extraction import KneePCA


def data():
    return make_blobs(n_samples=100, n_features=30, centers=5, random_state=1)


class KneePCATest(unittest.TestCase):
    def test_fits_with_no_error(self):
        X, _ = data()
        KneePCA().fit(X)

    def test_fit_transform_limits_dims(self):
        X, _ = data()
        X_trans = KneePCA().fit_transform(X)
        self.assertLess(X_trans.shape[1], X.shape[1])

    def test_fit_transform_preserves_all_observations(self):
        X, _ = data()
        X_trans = KneePCA().fit_transform(X)
        self.assertEqual(X.shape[0], X_trans.shape[0])

    def test_inverse_transform_preserves_shape(self):
        X, _ = data()
        pca = KneePCA()
        X_trans = pca.fit_transform(X)
        X_inv = pca.inverse_transform(X_trans)
        self.assertEqual(X_inv.shape, X.shape)

    def test_inverse_transform_preserves_most_values(self):
        X, _ = data()
        pca = KneePCA()
        X_trans = pca.fit_transform(X)
        X_inv = pca.inverse_transform(X_trans)
        # rows should be at least correlated a bit
        d = cdist(X_inv, X, metric='correlation')
        d = d[np.eye(d.shape[0], dtype=bool)]
        self.assertLess(d.max(), 0.2)
    
    def test_there_is_not_much_data_loss(self):
        X, _ = make_blobs(n_samples=100, n_features=3, centers=5,
                          random_state=0)
        pca = KneePCA()
        X_trans = pca.fit_transform(X)
        X_inv = pca.inverse_transform(X_trans)
        np.testing.assert_array_almost_equal(X, X_inv, decimal=3)

    def test_refit_inverse_transform_preserves_shape(self):
        X, _ = data()
        pca = KneePCA(refit=True)
        X_trans = pca.fit_transform(X)
        X_inv = pca.inverse_transform(X_trans)
        self.assertEqual(X_inv.shape, X.shape)

    def test_refit_does_not_introduce_data_loss(self):
        X, _ = make_blobs(n_samples=100, n_features=3, centers=5,
                          random_state=0)
        pca = KneePCA(refit=True)
        X_trans = pca.fit_transform(X)
        X_inv = pca.inverse_transform(X_trans)
        np.testing.assert_array_almost_equal(X, X_inv, decimal=3)


if __name__ == '__main__':
    unittest.main()
