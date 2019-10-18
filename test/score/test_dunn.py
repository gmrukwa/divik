import unittest

import numpy as np

import divik.score
from divik.distance import ScipyDistance, KnownMetric


class TestDunn(unittest.TestCase):
    def test_computes_inter_to_intracluster_distances_rate(self):
        data = np.array([[1], [3], [4], [6]])
        centroids = np.array([[2], [5]])
        labels = np.array([1, 1, 2, 2], dtype=int)
        distance = ScipyDistance(KnownMetric.euclidean)
        dunn = divik.score.dunn(data, labels, centroids, distance)
        self.assertAlmostEqual(dunn, 3.)
