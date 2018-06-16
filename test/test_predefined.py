import unittest

import numpy as np

import spdivik.predefined as pre


class TestProteomic(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.data = np.random.randn(300, 100)
        # vary some features
        idx = np.random.randint(0, 100, 20)
        self.data[:, tuple(idx)] *= 3
        # upregulate some features
        idx = np.random.randint(0, 100, 80)
        self.data[:, tuple(idx)] += 100
        # separate a group
        idx = np.random.randint(0, 300, 100)
        self.data[tuple(idx), :] += 100

    def test_constructs_runnable_pipeline(self):
        divik = pre.proteomic(minimal_split_segment=10)
        some_result = divik(self.data)
        self.assertIsNotNone(some_result)

    def test_splits_test_data_into_two_topmost_groups(self):
        divik = pre.proteomic(minimal_split_segment=10)
        some_result = divik(self.data)
        self.assertEqual(len(some_result.subregions), 2)

    def test_preserves_information_about_noise_filter(self):
        divik = pre.proteomic(minimal_split_segment=10)
        some_result = divik(self.data)
        self.assertIn('amplitude', some_result.thresholds)
        self.assertIn('amplitude', some_result.filters)
        self.assertNotIn('amplitude', some_result.subregions[0].thresholds)

    def test_scores_segmentation(self):
        divik = pre.proteomic(minimal_split_segment=10)
        some_result = divik(self.data)
        self.assertFalse(np.isnan(some_result.quality))
