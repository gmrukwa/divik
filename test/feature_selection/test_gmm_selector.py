import unittest

import numpy as np
import numpy.testing as npt
from sklearn.metrics import accuracy_score

import divik.feature_selection as fs


class GMMSelectorTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.labels = np.concatenate(
            [30 * [0] + 20 * [1] + 30 * [2] + 40 * [3]])
        self.data = self.labels * 5 + np.random.randn(*self.labels.shape)
        self.data = self.data.reshape((1, -1))

    def test_selects_high_component_by_default(self):
        selector = fs.GMMSelector('mean').fit(self.data)
        npt.assert_array_equal(selector.selected_,
                               self.labels == self.labels.max())

    def test_selects_low_component_when_set(self):
        selector = fs.GMMSelector('mean', preserve_high=False).fit(self.data)
        npt.assert_array_equal(selector.selected_,
                               self.labels == self.labels.min())

    def test_that_0_candidates_preserves_all_the_features(self):
        selector = fs.GMMSelector('mean', n_candidates=0).fit(self.data)
        npt.assert_array_equal(selector.selected_, True)

    def test_can_preserve_more_components(self):
        selector = fs.GMMSelector('mean', n_candidates=-1).fit(self.data)
        self.assertGreaterEqual(
            accuracy_score(self.labels >= 2, selector.selected_), 0.99)
        selector = fs.GMMSelector('mean', n_candidates=2).fit(self.data)
        self.assertGreaterEqual(
            accuracy_score(self.labels >= 2, selector.selected_), 0.99)
        selector = fs.GMMSelector('mean', n_candidates=1).fit(self.data)
        self.assertGreaterEqual(
            accuracy_score(self.labels >= 1, selector.selected_), 0.99)
        selector = fs.GMMSelector('mean', n_candidates=1, preserve_high=False)
        selector.fit(self.data)
        self.assertGreaterEqual(
            accuracy_score(self.labels < 3, selector.selected_), 0.99)

    def test_fails_log_with_negative_features(self):
        self.data[0, 1] = -1
        selector = fs.GMMSelector('mean', use_log=True)
        with self.assertRaises(ValueError):
            selector.fit(self.data)

    def test_works_for_mean_and_var_only(self):
        fs.GMMSelector('mean')
        fs.GMMSelector('var')
        with self.assertRaises(ValueError):
            fs.GMMSelector('yolo')

    def test_preserves_min_features_precisely_or_rate(self):
        selector = fs.GMMSelector('mean', min_features=50).fit(self.data)
        self.assertGreaterEqual(selector.selected_.sum(), 50)
        selector = fs.GMMSelector('mean', min_features_rate=0.5).fit(self.data)
        self.assertGreaterEqual(selector.selected_.sum(),
                                0.5 * self.labels.size)
