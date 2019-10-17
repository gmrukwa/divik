import unittest

import numpy as np

import divik.rejection as rj


def make_segmentation(partition: np.ndarray):
    return partition, np.ones((np.unique(partition).size, 1)), 1.234


class SizeTest(unittest.TestCase):
    def test_rejects_clusters_with_small_size(self):
        segmentation = make_segmentation([1, 2, 2])
        self.assertTrue(rj.reject_if_clusters_smaller_than(segmentation, 2))

    def test_allows_clusters_with_sufficient_size(self):
        segmentation = make_segmentation([1, 2, 2])
        self.assertFalse(rj.reject_if_clusters_smaller_than(segmentation, 1))
        segmentation = make_segmentation([1, 1, 2, 2, 3, 3, 3])
        self.assertFalse(rj.reject_if_clusters_smaller_than(segmentation, 2))
