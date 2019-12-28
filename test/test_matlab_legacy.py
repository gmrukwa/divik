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


def blobs(n):
    def some_blobs():
        return make_blobs(n_samples=500, centers=np.reshape(n, (1, -1)),
                          n_features=1, random_state=1)[0].ravel()
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
        blobs([-1.559, 0.637, 1.111, 2.043, 4.223]),
        blobs([-1.746, -1.366, 0.852, 1.023, 1.920, 2.860, 4.467]),
        blobs([-2.388, -1.681, -1.495, 0.841, 0.920, 2.202, 2.914, 4.396,
               7.694, 9.292]),
        blobs([-2.684, -1.744, -1.374, 0.412, 0.902, 1.045, 1.273, 2.259,
               2.797, 4.245, 6.044, 7.879, 9.041]),
        blobs([-9.824, -8.520, -8.313, -2.030, -1.464, -1.373, 0.406, 0.972,
               0.975, 1.127, 2.223, 2.663, 4.200, 5.368, 7.642, 8.644, 9.215]),
        blobs([-9.552, -8.903, -8.250, -2.686, -1.607, -1.437, 0.530, 1.008,
               1.203, 1.433, 2.030, 3.032, 3.877, 5.457, 5.991, 6.604, 7.323,
               8.006, 8.307, 9.146]),
        blobs([-5.578, -4.302, 0.873, 3.913, 4.375]),
        blobs([-5.510, -4.285, -1.618, 0.883, 3.933, 4.216, 9.586]),
        blobs([-5.421, -4.277, -2.010, -1.621, -0.376, 0.975, 3.481, 4.051,
               4.269, 9.495]),
        blobs([-5.587, -3.929, -3.234, -2.294, -1.731, -1.205, -0.619, 1.102,
               3.666, 3.700, 4.252, 4.574, 9.528]),
        blobs([-8.813, -6.537, -5.505, -4.050, -3.342, -2.193, -1.866, -1.591,
               -1.274, -0.554, 1.170, 3.624, 3.657, 4.306, 4.384, 4.627,
               9.393]),
        blobs([-9.079, -6.430, -6.385, -5.596, -3.761, -3.541, -2.178, -2.082,
               -1.494, -1.380, -0.624, 0.737, 0.739, 0.975, 3.643, 3.678,
               4.552, 4.646, 4.753, 9.522]),
    ]])
    def test_consistency(self, _, values):
        native = ml.find_thresholds_native(values)
        mcr = ml.find_thresholds_mcr(values)
        range_ = values.max() - values.min()
        native_ = (native - values.min()) / range_
        mcr_ = (mcr - values.min()) / range_
        npt.assert_almost_equal(native_, mcr_, decimal=3)
