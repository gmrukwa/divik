import unittest
from unittest.mock import patch

import numpy as np

import spdivik.kmeans as km

import spdivik.distance as dist


data = np.array([
    [1, 1, 1, 1],
    [2, 4, 2, 2],
    [1.9, 4.2, 1.9, 1.9],
    [2, 2, 2, 2],
    [1.1, 0.8, 1.1, 1.1],
    [1000, 490231, -412342, -7012]
])


class ExtremeInitializationTest(unittest.TestCase):
    def setUp(self):
        self.number_of_clusters = 2
        self.distance = dist.ScipyDistance(dist.KnownMetric.euclidean)
        self.initialize = km.ExtremeInitialization(self.distance)

    def test_uses_given_distance(self):
        with patch.object(dist.ScipyDistance, '__call__') as mock:
            self.initialize(data, self.number_of_clusters)
            self.assertGreater(mock.call_count, 0)

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


class LabelingTest(unittest.TestCase):
    def setUp(self):
        self.label = km.Labeling(dist.ScipyDistance(dist.KnownMetric.euclidean))
        self.centroids = np.array([
            [1, 1, 1, 1],
            [1000, 500000, -400000, -7000]
        ])

    def test_assigns_observations_to_closest_centroid(self):
        labels = self.label(data, self.centroids)
        np.testing.assert_equal(labels[:-1], 0)
        self.assertEqual(labels[-1], 1)

    def test_throws_with_no_centroids(self):
        with self.assertRaises(ValueError):
            self.label(data, np.empty((0, data.shape[1])))

    def test_throws_on_centroids_and_data_dimensionality_mismatch(self):
        smaller = data[:, :-1]
        with self.assertRaises(ValueError):
            self.label(smaller, self.centroids)
