import unittest
from unittest.mock import create_autospec, patch

import numpy as np

from divik.cluster._kmeans import _initialization as km
from test.cluster.kmeans import data


def measure(func):
    return create_autospec(func, side_effect=func)


class ExtremeInitializationTest(unittest.TestCase):
    def setUp(self):
        self.number_of_clusters = 2
        self.initialize = km.ExtremeInitialization('euclidean')

    def test_centroids_have_the_same_number_of_features_as_data(self):
        centroids = self.initialize(data, self.number_of_clusters)
        self.assertEqual(centroids.shape[1], data.shape[1])

    def test_number_of_centroids_is_preserved(self):
        centroids = self.initialize(data, self.number_of_clusters)
        self.assertEqual(centroids.shape[0], self.number_of_clusters)

    def test_works_without_error_for_ill_conditioned_problems(self):
        self.initialize(data[0:3], self.number_of_clusters)

    def test_throws_when_required_more_centroids_than_data(self):
        with self.assertRaises(ValueError):
            self.initialize(data[0:self.number_of_clusters-1],
                            self.number_of_clusters)

    def test_first_centroid_is_furthest_from_all(self):
        centroid = self.initialize(data, 1)
        np.testing.assert_equal(centroid, data[-1].reshape(1, -1))

    def test_next_centroid_is_furthest_from_already_found(self):
        centroids = self.initialize(data, 3)
        np.testing.assert_equal(centroids[1], data[4])
        np.testing.assert_equal(centroids[2], data[2])


# 99-th percentile should be approximated to max
class PercentileInitializationTest(ExtremeInitializationTest):
    def setUp(self):
        self.number_of_clusters = 2
        self.initialize = km.PercentileInitialization('euclidean')


class KDTreePercentileInitializationTest(unittest.TestCase):
    def setUp(self):
        self.number_of_clusters = 2
        self.initialize = km.KDTreePercentileInitialization(
            'euclidean', 0.1, 99.)

    def test_works_for_sample_data(self):
        self.initialize(data, self.number_of_clusters)
