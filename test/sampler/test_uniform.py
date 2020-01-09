import unittest

from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs

from divik.sampler._uniform_sampler import UniformSampler, UniformPCASampler


def data():
    return make_blobs(n_samples=1000, n_features=30, centers=5, random_state=1)


class UniformSamplerTest(unittest.TestCase):
    def test_keeps_shapes(self):
        X, _ = data()
        sampler = UniformSampler().fit(X)
        self.assertEqual(sampler.get_sample(0).shape, X.shape)
        sampler = UniformSampler(n_rows=10).fit(X)
        self.assertEqual(sampler.get_sample(0).shape, (10, X.shape[1]))

    def test_keeps_limits(self):
        X, _ = data()
        sampler = UniformSampler().fit(X)
        X_rand = sampler.get_sample(0)
        self.assertTrue((X.min() <= X_rand.min()).all())
        self.assertTrue((X.max() >= X_rand.max()).all())


class UniformPCASamplerTest(unittest.TestCase):
    def test_keeps_shapes(self):
        X, _ = data()
        sampler = UniformPCASampler().fit(X)
        self.assertEqual(sampler.get_sample(0).shape, X.shape)
        sampler = UniformPCASampler(n_rows=10).fit(X)
        self.assertEqual(sampler.get_sample(0).shape, (10, X.shape[1]))

    def test_keeps_correlation(self):
        X, _ = data()
        sampler = UniformPCASampler().fit(X)
        X_rand = sampler.get_sample(0)
        d = float(cdist([X.mean(axis=0)], [X_rand.mean(axis=0)],
                        metric='correlation'))
        self.assertLess(d, 0.15)


if __name__ == '__main__':
    unittest.main()
