import unittest

from parameterized import parameterized
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from divik.cluster._kmeans._core import KMeans
from divik.cluster._kmeans._gap import GAPSearch


def data(n_clusters):
    return make_blobs(n_samples=10000, n_features=2, centers=n_clusters,
                      random_state=0)


class GAPSearchTest(unittest.TestCase):
    @parameterized.expand([
        ("{}_clusters".format(k), k) for k in [1, 2, 3, 4, 7, 8]
    ])
    def test_works_with_gap(self, _, n_clusters):
        X, y = data(n_clusters)
        single_kmeans = KMeans(n_clusters=2)
        kmeans = GAPSearch(single_kmeans, max_clusters=10,
                           sample_size=10000).fit(X)
        rand = adjusted_rand_score(y, kmeans.labels_)
        # allow for misidentification of 1 cluster
        self.assertGreaterEqual(kmeans.n_clusters_ + 1, n_clusters)
        self.assertLessEqual(kmeans.n_clusters_ - 1, n_clusters)
        self.assertGreater(rand, 0.75)

    @parameterized.expand([
        ("{}_clusters".format(k), k) for k in [1, 2, 3, 4, 7, 8]
    ])
    def test_works_with_sampled_gap(self, _, n_clusters):
        X, y = data(n_clusters)
        single_kmeans = KMeans(n_clusters=2)
        kmeans = GAPSearch(single_kmeans, max_clusters=10).fit(X)
        rand = adjusted_rand_score(y, kmeans.labels_)
        # allow for misidentification of 1 cluster
        self.assertGreaterEqual(kmeans.n_clusters_ + 1, n_clusters)
        self.assertLessEqual(kmeans.n_clusters_ - 1, n_clusters)
        self.assertGreater(rand, 0.75)

    def test_works_with_unfit_removal(self):
        n_clusters = 3
        X, y = data(n_clusters)
        single_kmeans = KMeans(n_clusters=2)
        kmeans = GAPSearch(
            single_kmeans, max_clusters=10, drop_unfit=True).fit(X)
        rand = adjusted_rand_score(y, kmeans.labels_)
        # allow for misidentification of 1 cluster
        self.assertGreaterEqual(kmeans.n_clusters_ + 1, n_clusters)
        self.assertLessEqual(kmeans.n_clusters_ - 1, n_clusters)
        self.assertGreater(rand, 0.75)
        self.assertIsNone(kmeans.estimators_)


if __name__ == '__main__':
    unittest.main()
