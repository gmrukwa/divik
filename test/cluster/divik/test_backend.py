import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt

import divik.cluster._divik._backend as dv


def returns(value):
    return MagicMock(return_value=value)


def returns_many(values):
    return MagicMock(side_effect=values)


def allow_no_feature(dataset: np.ndarray):
    return np.zeros((1, dataset.shape[1])), np.inf


def allow_almost_all_features(dataset: np.ndarray):
    return np.hstack([np.zeros((1, 1)), np.ones((1, dataset.shape[1] - 1))]), np.inf


DUMMY_DATA = np.arange(100).reshape(10, 10)
SELECT_ALL = np.ones((10,), dtype=bool)


class RecursiveSelectionTest(unittest.TestCase):
    def setUp(self):
        self.selection = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
                                  dtype=bool)
        self.partition = np.array([1, 2, 2, 1], dtype=int)

    def test_selects_subset_of_elements_by_cluster_number(self):
        npt.assert_equal(
            dv._recursive_selection(self.selection, self.partition, 1),
            np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=bool))
        npt.assert_equal(
            dv._recursive_selection(self.selection, self.partition, 2),
            np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0], dtype=bool))

    def test_selects_nothing_for_nonexistent_label(self):
        npt.assert_equal(
            dv._recursive_selection(self.selection, self.partition, 3),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool))
