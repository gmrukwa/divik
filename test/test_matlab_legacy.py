import unittest

import numpy as np

import divik._matlab_legacy as ml


class TestFindThresholds(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_separates_two_components(self):
        first = np.random.randn(100)
        second = 100 + 2 * np.random.randn(100)
        values = np.hstack((first, second))
        thresholds = ml.find_thresholds(values, max_components=2)
        self.assertEqual(len(thresholds), 1)
        self.assertLess(thresholds[0], 100)
        self.assertGreater(thresholds[0], 0)

    def test_separates_multiple_components(self):
        first = np.random.randn(1000)
        second = 100 + 2 * np.random.randn(1000)
        third = -second
        values = np.hstack((first, -first, second, third))
        thresholds = ml.find_thresholds(values, max_components=3)
        self.assertEqual(len(thresholds), 2)
        self.assertLess(thresholds[0], 0)
        self.assertGreater(thresholds[0], -100)
        self.assertLess(thresholds[1], 100)
        self.assertGreater(thresholds[1], 0)
