import unittest

import numpy as np
from sklearn.datasets import make_blobs

import divik.cluster as km
import divik.score as sc


def dummy_kmeans(labels, cluster_centers):
    kmeans = km.KMeans(
        n_clusters=len(cluster_centers),
        distance="euclidean",
        init="extreme",
        max_iter=10,
    )
    kmeans.labels_ = labels
    kmeans.cluster_centers_ = cluster_centers
    return kmeans


class TestGap(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(n_samples=10000, n_features=2, centers=3, random_state=0)
        self.kmeans_3 = km.KMeans(n_clusters=3).fit(self.X)
        self.kmeans_7 = km.KMeans(n_clusters=7).fit(self.X)

    def test_computes_score(self):
        score = sc.gap(self.X, self.kmeans_3)
        assert not np.isnan(score)

    def test_good_labeling_has_greater_score(self):
        better = sc.gap(self.X, self.kmeans_3)
        worse = sc.gap(self.X, self.kmeans_7)
        assert better > worse

    def test_returns_std_if_requested(self):
        gap, std = sc.gap(self.X, self.kmeans_3, return_deviation=True)
        assert gap is not None
        assert std is not None

    def test_works_with_sklearn(self):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, max_iter=2).fit(self.X)
        score = sc.gap(self.X, kmeans)
        assert not np.isnan(score)
