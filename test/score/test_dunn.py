import unittest

import numpy as np
from sklearn.datasets import make_blobs

import divik.cluster as km
from divik.score._dunn import (
    _inter_centroid,
    _inter_closest,
    _intra_avg,
    _intra_furthest,
    dunn,
    sampled_dunn,
)


class DummyKMeans:
    n_clusters = 2
    cluster_centers_ = np.array([[2], [5]])
    labels_ = np.array([0, 0, 1, 1], dtype=int)
    distance = 'euclidean'


class TestDunnDistances(unittest.TestCase):
    def test_inter_centroid(self):
        data = np.array([[1], [3], [4], [6]])
        dst = _inter_centroid(DummyKMeans(), data)
        self.assertAlmostEqual(dst, 3.)

    def test_inter_closest(self):
        data = np.array([[1], [3], [4], [6]])
        dst = _inter_closest(DummyKMeans(), data)
        self.assertAlmostEqual(dst, 1.)

    def test_intra_avg(self):
        data = np.array([[1], [3], [4], [6]])
        dst = _intra_avg(DummyKMeans(), data)
        self.assertAlmostEqual(dst, 1.)

    def test_intra_furthest(self):
        data = np.array([[1], [3], [4], [6]])
        dst = _intra_furthest(DummyKMeans(), data)
        self.assertAlmostEqual(dst, 2.)


class TestDunn(unittest.TestCase):
    def test_computes_inter_to_intracluster_distances_rate(self):
        data = np.array([[1], [3], [4], [6]])
        dunn_ = dunn(DummyKMeans(), data)
        self.assertAlmostEqual(dunn_, 3.)

    def test_works_with_other_distances(self):
        data = np.array([[1], [3], [4], [6]])
        dunn_ = dunn(DummyKMeans(), data, inter='closest', intra='furthest')
        self.assertAlmostEqual(dunn_, .5)


class TestSamplingDunn(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(n_samples=10000, n_features=2, centers=3,
                               random_state=0, cluster_std=0.1)
        self.kmeans_3 = km.KMeans(n_clusters=3).fit(self.X)
        self.kmeans_7 = km.KMeans(n_clusters=7).fit(self.X)

    def test_computes_score(self):
        score = sampled_dunn(self.kmeans_3, self.X)
        self.assertFalse(np.isnan(score))

    def test_good_labeling_has_top_score(self):
        kmeans = [km.KMeans(n_clusters=k, init='kdtree').fit(self.X)
                  for k in range(2, 11)]
        dunn_ = [sampled_dunn(mdl, self.X) for mdl in kmeans]
        best = int(np.argmax(dunn_))
        self.assertEqual(kmeans[best].n_clusters, self.kmeans_3.n_clusters)
