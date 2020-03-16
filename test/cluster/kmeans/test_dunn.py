import unittest

from parameterized import parameterized
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from divik.cluster._kmeans._core import KMeans
from divik.cluster._kmeans._dunn import DunnSearch


# if the std is too big, it won't work with high number of clusters,
# because the 20x20 grid is quite small for 10 000 points
def data(n_clusters):
    return make_blobs(n_samples=10000, n_features=2, centers=n_clusters,
                      random_state=0, cluster_std=0.1)


class DunnSearchTest(unittest.TestCase):
    # Simplified Dunn favorizes small number of clusters.
    @parameterized.expand([
        ("{}_clusters".format(k), k) for k in [2, 3, 4]
    ])
    def test_works_with_simplified_dunn(self, _, n_clusters):
        X, y = data(n_clusters)
        single_kmeans = KMeans(n_clusters=2, init='kdtree')
        kmeans = DunnSearch(single_kmeans, max_clusters=10).fit(X)
        rand = adjusted_rand_score(y, kmeans.labels_)
        self.assertEqual(n_clusters, kmeans.n_clusters_)
        self.assertGreater(rand, 0.75)

    @parameterized.expand([
        ("{}_clusters".format(k), k) for k in range(2, 9)
    ])
    def test_works_with_full_exact_dunn(self, _, n_clusters):
        X, y = data(n_clusters)
        single_kmeans = KMeans(n_clusters=2, init='kdtree')
        kmeans = DunnSearch(single_kmeans, max_clusters=15,
                            inter='closest', intra='furthest',
                            ).fit(X)
        rand = adjusted_rand_score(y, kmeans.labels_)
        self.assertEqual(n_clusters, kmeans.n_clusters_)
        self.assertGreater(rand, 0.75)

    @parameterized.expand([
        ("{}_clusters".format(k), k) for k in range(2, 9)
    ])
    def test_works_with_sampled_exact_dunn(self, _, n_clusters):
        X, y = data(n_clusters)
        single_kmeans = KMeans(n_clusters=2, init='kdtree')
        kmeans = DunnSearch(single_kmeans, max_clusters=15, method='sampled',
                            inter='closest', intra='furthest', n_trials=10,
                            ).fit(X)
        rand = adjusted_rand_score(y, kmeans.labels_)
        self.assertEqual(n_clusters, kmeans.n_clusters_)
        self.assertGreater(rand, 0.75)
    
    def test_works_with_unfit_removal(self):
        n_clusters = 3
        X, y = data(n_clusters)
        single_kmeans = KMeans(n_clusters=2)
        kmeans = DunnSearch(
            single_kmeans, max_clusters=10, drop_unfit=True).fit(X)
        rand = adjusted_rand_score(y, kmeans.labels_)
        self.assertEqual(n_clusters, kmeans.n_clusters_)
        self.assertGreater(rand, 0.75)
        self.assertIsNone(kmeans.estimators_)


if __name__ == '__main__':
    unittest.main()
