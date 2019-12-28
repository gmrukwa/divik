import unittest
from parameterized import parameterized

import numpy as np
import numpy.testing as npt
from sklearn.datasets import make_blobs

import divik._matlab_legacy as ml


def no_threshold():
    return np.random.randn(500)


def single_th_between_0_and_100():
    first = np.random.randn(100)
    second = 100 + 2 * np.random.randn(100)
    values = np.hstack((first, second))
    return values


def two_ths_between_minus_100_0_and_100():
    first = np.random.randn(1000)
    second = 100 + 2 * np.random.randn(1000)
    third = -second
    values = np.hstack((first, -first, second, third))
    return values


def blobs(n, seed=0):
    def some_blobs():
        return make_blobs(n_samples=500, centers=n, n_features=1,
                          random_state=seed)[0].ravel()
    return some_blobs


class TestFindThresholdsMcr(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_separates_two_components(self):
        values = single_th_between_0_and_100()
        thresholds = ml.find_thresholds_mcr(values, max_components=2)
        self.assertEqual(len(thresholds), 1)
        self.assertLess(thresholds[0], 100)
        self.assertGreater(thresholds[0], 0)

    def test_separates_multiple_components(self):
        values = two_ths_between_minus_100_0_and_100()
        thresholds = ml.find_thresholds_mcr(values, max_components=3)
        self.assertEqual(len(thresholds), 2)
        self.assertLess(thresholds[0], 0)
        self.assertGreater(thresholds[0], -100)
        self.assertLess(thresholds[1], 100)
        self.assertGreater(thresholds[1], 0)


class TestFindThresholdsNative(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_separates_two_components(self):
        values = single_th_between_0_and_100()
        thresholds = ml.find_thresholds_native(values, max_components=2)
        self.assertEqual(len(thresholds), 1)
        self.assertLess(thresholds[0], 100)
        self.assertGreater(thresholds[0], 0)

    def test_separates_multiple_components(self):
        values = two_ths_between_minus_100_0_and_100()
        thresholds = ml.find_thresholds_native(values, max_components=3)
        self.assertEqual(len(thresholds), 2)
        self.assertLess(thresholds[0], 0)
        self.assertGreater(thresholds[0], -100)
        self.assertLess(thresholds[1], 100)
        self.assertGreater(thresholds[1], 0)


class TestFindThresholdsConsistency(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    @parameterized.expand([(f.__name__, f()) for f in [
        no_threshold,
        single_th_between_0_and_100,
        two_ths_between_minus_100_0_and_100,
        blobs(5),
        blobs(7),
        blobs(10),
        blobs(13),
        blobs(17),
        blobs(20),
        blobs(5, seed=123),
        blobs(7, seed=123),
        blobs(10, seed=123),
        blobs(13, seed=123),
        blobs(17, seed=123),
        blobs(20, seed=123),
    ]])
    def test_consistency(self, _, values):
        native = ml.find_thresholds_native(values)
        mcr = ml.find_thresholds_mcr(values)
        npt.assert_almost_equal(native, mcr, decimal=3)
