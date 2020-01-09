import os
import unittest
from parameterized import parameterized

import numpy as np
import numpy.testing as npt
from sklearn.datasets import make_blobs

import divik._matlab_legacy as ml


def no_threshold():
    np.random.seed(0)
    return np.random.randn(500)


def single_th_between_0_and_100():
    np.random.seed(0)
    first = np.random.randn(100)
    second = 100 + 2 * np.random.randn(100)
    values = np.hstack((first, second))
    return values


def two_ths_between_minus_100_0_and_100():
    np.random.seed(0)
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


class TestFindThresholdsNative(unittest.TestCase):
    def test_separates_two_components(self):
        values = single_th_between_0_and_100()
        thresholds = ml.find_thresholds(values, max_components=2)
        self.assertEqual(len(thresholds), 1)
        self.assertLess(thresholds[0], 100)
        self.assertGreater(thresholds[0], 0)

    def test_separates_multiple_components(self):
        values = two_ths_between_minus_100_0_and_100()
        thresholds = ml.find_thresholds(values, max_components=3)
        self.assertEqual(len(thresholds), 2)
        self.assertLess(thresholds[0], 0)
        self.assertGreater(thresholds[0], -100)
        self.assertLess(thresholds[1], 100)
        self.assertGreater(thresholds[1], 0)


quick_cases = [
    no_threshold,
    single_th_between_0_and_100,
    two_ths_between_minus_100_0_and_100,
]
slow_cases = [
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
]
enable_long = os.environ.get('ENABLE_SLOW_TESTS', 'False').lower() == 'true'

cases = quick_cases + slow_cases if enable_long else quick_cases
cases_expectations = [
    [1.9128136053649736],
    [32.861988912865336],
    [-33.82253469037943, 33.82253368238841],
    [-1.039095608844928, 3.616002058353841],
    [-0.3984068175747515, 3.4712354466624937],
    [-0.8047461812891425, 6.353972855207196],
    [-1.0066505540513182, 4.042247554972411, 7.535675286954712],
    [-5.141469013285568, -0.47571379978287176, 3.2771751168828303,
     6.293933851327623],
    [-5.259208899972172, -1.2918941569672793, 4.532732128486252],
    [-1.6103974380886266, 2.403528798683775],
    [-3.6031042466477325, 2.5082520815516993, 7.076781428728767],
    [-3.7913329830380516, 2.1936334536419064, 7.040122593178744],
    [1.85518201331314, 7.185332860428021],
    [-7.9103491430755355, -4.887751754416515, 1.8846776021326193,
     7.127370908336459],
    [-7.786305907571306, -5.29228495409376, 2.7848138019657256,
     7.2727379351231995],
]


class TestFindThresholdsConsistency(unittest.TestCase):
    @parameterized.expand(
        [(f.__name__, f(), e) for f, e in zip(cases, cases_expectations)])
    def test_consistency(self, _, values, expected):
        native = ml.find_thresholds(values)
        mcr = np.array(expected)
        range_ = values.max() - values.min()
        native_ = (native - values.min()) / range_
        mcr_ = (mcr - values.min()) / range_
        npt.assert_almost_equal(native_, mcr_, decimal=3)
