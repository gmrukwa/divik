import unittest
from unittest.mock import MagicMock, patch
from itertools import cycle

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


class CentroidRedefinitionTest(unittest.TestCase):
    def setUp(self):
        self.labeling = np.array([0, 0, 0, 1, 1])
        self.simple_data = np.array([
            [1, 2, 3],
            [3, 2, 1],
            [2, 2, 2],
            [10, 10, 10],
            [40, 40, 40]
        ])
        self.expected_centroids = np.array([
            [2, 2, 2],
            [25, 25, 25]
        ])

    def test_new_centroids_are_means_of_observations_in_cluster(self):
        centroids = km.redefine_centroids(self.simple_data, self.labeling)
        np.testing.assert_equal(centroids, self.expected_centroids)

    def test_throws_on_labels_and_data_length_mismatch(self):
        with self.assertRaises(ValueError):
            km.redefine_centroids(self.simple_data[:-1], self.labeling)


# noinspection PyTypeChecker
class KMeansTest(unittest.TestCase):
    def setUp(self):
        self.mocked_initial_centroids = data[[0, -1]]
        self.mock_initialization = MagicMock(
            return_value=self.mocked_initial_centroids)
        self.mocked_labels = np.hstack([
            np.zeros((data.shape[0] - 1,)),
            np.ones((1,))
        ])
        self.mock_labeling = MagicMock(side_effect=cycle([
            self.mocked_labels,
            self.mocked_labels + 3
        ]))
        self.kmeans = km.KMeans(labeling=self.mock_labeling,
                                initialize=self.mock_initialization,
                                number_of_iterations=3)

    def test_only_initializes_for_no_iterations(self):
        self.kmeans.number_of_iterations = 0
        with patch.object(km, "redefine_centroids") as mock_redefine:
            labeling, centroids = self.kmeans(data, 2)
        self.assertEqual(0, mock_redefine.call_count)
        self.assertEqual(1, self.mock_initialization.call_count)
        self.assertEqual(1, self.mock_labeling.call_count)
        np.testing.assert_equal(self.mocked_labels, labeling)
        np.testing.assert_equal(self.mocked_initial_centroids, centroids)

    def test_initializes_once(self):
        with patch.object(km, "redefine_centroids"):
            self.kmeans(data, 2)
        self.assertEqual(1, self.mock_initialization.call_count)

    def test_redefines_centroids_each_iteration(self):
        with patch.object(km, "redefine_centroids") as mock_redefine:
            self.kmeans(data, 2)
        self.assertEqual(self.kmeans.number_of_iterations,
                         mock_redefine.call_count)

    def test_recalculates_labels_on_init_and_each_iteration(self):
        with patch.object(km, "redefine_centroids"):
            self.kmeans(data, 2)
        self.assertEqual(self.kmeans.number_of_iterations + 1,
                         self.mock_labeling.call_count)

    def test_returns_final_labels_and_centroids(self):
        with patch.object(km, "redefine_centroids") as mock_redefine:
            mock_redefine.return_value = self.mocked_initial_centroids + 3
            labeling, centroids = self.kmeans(data, 2)
        np.testing.assert_equal(centroids, self.mocked_initial_centroids + 3)
        np.testing.assert_equal(labeling, self.mocked_labels + 3)

    def test_breaks_when_reaches_stability(self):
        constant_labels = self.mocked_labels
        self.mock_labeling.side_effect = None
        self.mock_labeling.return_value = constant_labels
        with patch.object(km, "redefine_centroids") as mock_redefine:
            self.kmeans(data, 2)
        self.assertEqual(1, mock_redefine.call_count)

    def test_throws_for_non_2d_and_non_matrix_data(self):
        with self.assertRaises(ValueError):
            self.kmeans(data.ravel(), 2)
        with self.assertRaises(ValueError):
            self.kmeans([[1, 2, 3], [4, 5, 6]], 2)


class KMeansIntegrationTest(unittest.TestCase):
    def test_segments_data_into_groups(self):
        np.random.seed(0)
        first_size = 200
        second_size = 100

        first = np.random.randn(first_size, 1)
        second = 100 + 3 * np.random.randn(second_size, 1)
        data = np.vstack([first, second])
        expected_labels = first_size * [0] + second_size * [1]

        distance = dist.ScipyDistance(dist.KnownMetric.euclidean)
        euclidean_labeling = km.Labeling(distance)
        kmeans = km.KMeans(labeling=euclidean_labeling,
                           initialize=km.ExtremeInitialization(distance),
                           number_of_iterations=100)

        labels, _ = kmeans(data, 2)

        labels_are_the_same = labels == expected_labels
        labels_are_opposite = labels != expected_labels
        labels_are_valid = np.logical_or(labels_are_the_same,
                                         labels_are_opposite)
        self.assertTrue(np.all(labels_are_valid))
