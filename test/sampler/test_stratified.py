import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

from divik.sampler._stratified_sampler import StratifiedSampler


def data():
    return make_blobs(n_samples=1000, n_features=30, centers=5, random_state=1)


class StratifiedSamplerTest(unittest.TestCase):
    def test_keeps_shapes(self):
        X, y = data()
        sampler = StratifiedSampler(n_rows=10).fit(X, y)
        self.assertEqual(sampler.get_sample(0).shape, (10, X.shape[1]))

    def test_keeps_proportion(self):
        X, y = data()
        sampler = StratifiedSampler().fit(X, y)
        X_rand = sampler.get_sample(0)
        # Rows should exist in X, so 1 neighbor is just exact match.
        y_rand = KNeighborsClassifier(n_neighbors=1).fit(X, y).predict(X_rand)
        _, orig_count = np.unique(y, return_counts=True)
        _, rand_count = np.unique(y_rand, return_counts=True)
        orig_prop = orig_count / np.linalg.norm(orig_count)
        rand_prop = rand_count / np.linalg.norm(rand_count)
        # This is equivalent to up to 1% of samples moved between labels
        self.assertLess(np.abs(orig_prop - rand_prop).max(), 0.01)

    def test_samples_exact_rows(self):
        X, y = data()
        sampler = StratifiedSampler().fit(X, y)
        X_rand = sampler.get_sample(0)
        for row in X_rand:
            self.assertTrue((row[np.newaxis, :] == X).all(axis=1).any(axis=0))


if __name__ == '__main__':
    unittest.main()
