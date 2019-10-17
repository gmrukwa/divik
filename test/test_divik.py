import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt

import divik.divik as dv
import divik.feature_selection as fs


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


class DivikBackendTest(unittest.TestCase):
    def test_calls_selection(self):
        with patch.object(fs, fs.select_sequentially.__name__,
                          new=returns((MagicMock,) * 3)) as selector:
            dv._divik_backend(data=DUMMY_DATA, selection=SELECT_ALL,
                              split=MagicMock(),
                              feature_selectors=[],
                              stop_condition=returns(True),
                              rejection_conditions=[],
                              report=MagicMock(),
                              min_features_percentage=.05,
                              prefiltering_stop_condition=lambda x: False)
        self.assertEqual(1, selector.call_count)
        self.assertSequenceEqual([], selector.call_args[0][0])
        self.assertEqual(.05, selector.call_args[0][2])

    def test_calls_stop_condition(self):
        stop_condition = returns(True)
        dv._divik_backend(data=DUMMY_DATA, selection=SELECT_ALL,
                          split=MagicMock(),
                          feature_selectors=[],
                          stop_condition=stop_condition,
                          rejection_conditions=[],
                          report=MagicMock(),
                          min_features_percentage=.05,
                          prefiltering_stop_condition=lambda x: False)
        self.assertEqual(1, stop_condition.call_count)

    def test_ends_when_stop_condition(self):
        stop_condition = returns(True)
        split = returns((MagicMock(),) * 3)
        tree = dv._divik_backend(data=DUMMY_DATA, selection=SELECT_ALL,
                                 split=split,
                                 feature_selectors=[],
                                 stop_condition=stop_condition,
                                 rejection_conditions=[],
                                 report=MagicMock(),
                                 min_features_percentage=.05,
                                 prefiltering_stop_condition=lambda x: False)
        self.assertEqual(0, split.call_count)
        self.assertIsNone(tree)

    def test_calls_split_when_not_stop_condition(self):
        stop_condition = returns_many([False, True, True])
        split = returns((MagicMock(),) * 3)
        reporter = MagicMock()
        with patch.object(np, np.unique.__name__, new=returns([1, 2])):
            tree = dv._divik_backend(data=DUMMY_DATA, selection=SELECT_ALL,
                                     split=split,
                                     feature_selectors=[],
                                     stop_condition=stop_condition,
                                     rejection_conditions=[],
                                     report=reporter,
                                     min_features_percentage=.05,
                                     prefiltering_stop_condition=lambda x: False)
        self.assertIsNotNone(tree)
        self.assertEqual(1, reporter.recurring.call_count)
        self.assertEqual(3, stop_condition.call_count)
        self.assertEqual(1, split.call_count)

    def test_rejects_unmatched_segmentation(self):
        stop_condition = returns_many([False, True, True])
        split = returns((MagicMock(),) * 3)
        tree = dv._divik_backend(data=DUMMY_DATA, selection=SELECT_ALL,
                                 split=split,
                                 feature_selectors=[],
                                 stop_condition=stop_condition,
                                 rejection_conditions=[returns(True)],
                                 report=MagicMock(),
                                 min_features_percentage=.05,
                                 prefiltering_stop_condition=lambda x: False)
        self.assertIsNone(tree)
        self.assertEqual(1, stop_condition.call_count)
        self.assertEqual(1, split.call_count)

    def test_forwards_all_the_options_unchanged(self):
        old = dv._divik_backend
        stop_condition = returns_many([False, True, True])
        split = returns((MagicMock(),) * 3)
        rejection_conditions = [returns(False)]
        report = MagicMock()
        def prefiltering(data):
            return False
        with patch.object(dv, dv._divik_backend.__name__) as mock, \
                patch.object(np, np.unique.__name__, new=returns([1, 2])), \
                patch.object(dv, dv._recursive_selection.__name__, new=returns(SELECT_ALL)):
            tree = old(data=DUMMY_DATA, selection=SELECT_ALL,
                       split=split,
                       feature_selectors=[],
                       stop_condition=stop_condition,
                       rejection_conditions=rejection_conditions,
                       report=report,
                       min_features_percentage=.15,
                       prefiltering_stop_condition=prefiltering)
        self.assertIsNotNone(tree)
        mock.assert_called_with(data=DUMMY_DATA, selection=SELECT_ALL,
                                split=split,
                                feature_selectors=[],
                                stop_condition=stop_condition,
                                rejection_conditions=rejection_conditions,
                                report=report,
                                min_features_percentage=.15,
                                prefiltering_stop_condition=prefiltering)

    def test_constructs_result(self):
        old = dv._divik_backend
        stop_condition = returns_many([False, True, True])
        split = returns((MagicMock(),) * 3)
        rejection_conditions = [returns(False)]
        report = MagicMock()
        with patch.object(dv, dv._divik_backend.__name__, new=returns(None)), \
             patch.object(np, np.unique.__name__, new=returns([1, 2])), \
             patch.object(dv, dv._recursive_selection.__name__, new=returns(SELECT_ALL)):
            tree = old(data=DUMMY_DATA, selection=SELECT_ALL,
                       split=split,
                       feature_selectors=[],
                       stop_condition=stop_condition,
                       rejection_conditions=rejection_conditions,
                       report=report,
                       min_features_percentage=.15,
                       prefiltering_stop_condition=lambda x: False)
        self.assertIsNotNone(tree)
        self.assertSequenceEqual([None, None], tree.subregions)
        self.assertSequenceEqual({}, tree.thresholds)
        self.assertSequenceEqual({}, tree.filters)
