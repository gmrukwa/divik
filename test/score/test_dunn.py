import unittest

import numpy as np

from divik._score._dunn import _dunn_backend


class TestDunn(unittest.TestCase):
    def test_computes_inter_to_intracluster_distances_rate(self):
        data = np.array([[1], [3], [4], [6]])
        centroids = np.array([[2], [5]])
        labels = np.array([1, 1, 2, 2], dtype=int)
        dunn = _dunn_backend(data, labels, centroids, 'euclidean')
        self.assertAlmostEqual(dunn, 3.)
