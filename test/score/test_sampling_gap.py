import unittest

import numpy as np
from sklearn.datasets import make_blobs

import divik.cluster as km
import divik.score as sc


class TestSamplingGap(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(n_samples=10000, n_features=2, centers=3,
                               random_state=0)
        self.kmeans_3 = km.KMeans(n_clusters=3).fit(self.X)
        self.kmeans_7 = km.KMeans(n_clusters=7).fit(self.X)

    def test_computes_score(self):
        score = sc.sampled_gap(self.X, self.kmeans_3)
        self.assertFalse(np.isnan(score))

    def test_good_labeling_has_greater_score(self):
        better = sc.sampled_gap(self.X, self.kmeans_3)
        worse = sc.sampled_gap(self.X, self.kmeans_7)
        self.assertGreater(better, worse)

    def test_returns_std_if_requested(self):
        gap, std = sc.sampled_gap(self.X, self.kmeans_3, return_deviation=True)
        self.assertIsNotNone(gap)
        self.assertIsNotNone(std)
