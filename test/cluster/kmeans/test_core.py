import unittest
from itertools import cycle
from unittest.mock import MagicMock, patch

import numpy as np

from divik.cluster import _kmeans as km
from divik.cluster._kmeans import _core as cc
from divik.cluster._kmeans._core import redefine_centroids
from test.cluster.kmeans import data


class LabelingTest(unittest.TestCase):
    def setUp(self):
        self.label = km.Labeling('euclidean')
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
        centroids = redefine_centroids(self.simple_data, self.labeling, [0, 1])
        np.testing.assert_equal(centroids, self.expected_centroids)

    def test_throws_on_labels_and_data_length_mismatch(self):
        with self.assertRaises(ValueError):
            redefine_centroids(self.simple_data[:-1], self.labeling, [0, 1])


class KMeansTest(unittest.TestCase):
    # noinspection PyTypeChecker
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
        self.kmeans = km._KMeans(labeling=self.mock_labeling,
                                 initialize=self.mock_initialization,
                                 number_of_iterations=3)

    def test_only_initializes_for_no_iterations(self):
        self.kmeans.number_of_iterations = 0
        with patch.object(km._core, "redefine_centroids") as mock_redefine:
            labeling, centroids = self.kmeans(data, 2)
        self.assertEqual(0, mock_redefine.call_count)
        self.assertEqual(1, self.mock_initialization.call_count)
        self.assertEqual(1, self.mock_labeling.call_count)
        np.testing.assert_equal(self.mocked_labels, labeling)
        np.testing.assert_equal(self.mocked_initial_centroids, centroids)

    def test_initializes_once(self):
        with patch.object(km._core, "redefine_centroids"):
            self.kmeans(data, 2)
        self.assertEqual(1, self.mock_initialization.call_count)

    def test_redefines_centroids_each_iteration(self):
        with patch.object(km._core, "redefine_centroids") as mock_redefine:
            self.kmeans(data, 2)
        self.assertEqual(self.kmeans.number_of_iterations,
                         mock_redefine.call_count)

    def test_recalculates_labels_on_init_and_each_iteration(self):
        with patch.object(km._core, "redefine_centroids"):
            self.kmeans(data, 2)
        self.assertEqual(self.kmeans.number_of_iterations + 1,
                         self.mock_labeling.call_count)

    def test_returns_final_labels_and_centroids(self):
        with patch.object(km._core, "redefine_centroids") as mock_redefine:
            mock_redefine.return_value = self.mocked_initial_centroids + 3
            labeling, centroids = self.kmeans(data, 2)
        np.testing.assert_equal(centroids, self.mocked_initial_centroids + 3)
        np.testing.assert_equal(labeling, self.mocked_labels + 3)

    def test_breaks_when_reaches_stability(self):
        constant_labels = self.mocked_labels
        self.mock_labeling.side_effect = None
        self.mock_labeling.return_value = constant_labels
        with patch.object(km._core, "redefine_centroids") as mock_redefine:
            self.kmeans(data, 2)
        self.assertEqual(1, mock_redefine.call_count)

    def test_throws_for_non_2d_and_non_matrix_data(self):
        with self.assertRaises(ValueError):
            self.kmeans(data.ravel(), 2)
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
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

        euclidean_labeling = km.Labeling('euclidean')
        kmeans = km._KMeans(labeling=euclidean_labeling,
                            initialize=cc.ExtremeInitialization('euclidean'),
                            number_of_iterations=100)

        labels, _ = kmeans(data, 2)

        labels_are_the_same = labels == expected_labels
        labels_are_opposite = labels != expected_labels
        labels_are_valid = np.logical_or(labels_are_the_same,
                                         labels_are_opposite)
        self.assertTrue(np.all(labels_are_valid))
