import unittest

from parameterized import parameterized
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from divik.cluster._kmeans._core import KMeans
from divik.cluster._kmeans._dunn import DunnSearch


def data(n_clusters):
    return make_blobs(n_samples=10000, n_features=2, centers=n_clusters,
                      random_state=0)


class DunnSearchTest(unittest.TestCase):
    # Dunn favorizes small number of clusters.
    @parameterized.expand([
        ("{}_clusters".format(k), k) for k in [2, 3, 4]
    ])
    def test_works_with_dunn(self, _, n_clusters):
        X, y = data(n_clusters)
        single_kmeans = KMeans(n_clusters=2)
        kmeans = DunnSearch(single_kmeans, max_clusters=10).fit(X)
        rand = adjusted_rand_score(y, kmeans.labels_)
        self.assertEqual(kmeans.n_clusters_, n_clusters)
        self.assertGreater(rand, 0.75)


if __name__ == '__main__':
    unittest.main()
