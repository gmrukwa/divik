import unittest

import numpy as np

import divik._feature_selection as fs


class HighAbundanceAndVarianceSelectorTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.labels = np.concatenate(
            [30 * [0] + 20 * [1] + 30 * [2] + 40 * [3]])
        self.data = np.vstack(100 * [self.labels * 10.])
        self.data += np.random.randn(*self.data.shape)
        sub = self.data[:, :-40]
        sub += 5 * np.random.randn(*sub.shape)

    def test_discards_low_abundance(self):
        selector = fs.HighAbundanceAndVarianceSelector().fit(self.data)
        TNR = (selector.selected_[self.labels == 0] == False).mean()
        self.assertGreaterEqual(TNR, 0.99)

    def test_discards_low_variance(self):
        selector = fs.HighAbundanceAndVarianceSelector().fit(self.data)
        TNR = (selector.selected_[self.labels == 3] == False).mean()
        self.assertGreaterEqual(TNR, 0.99)

    def test_passes_informative(self):
        selector = fs.HighAbundanceAndVarianceSelector().fit(self.data)
        expected = np.logical_or(self.labels == 1, self.labels == 2)
        TPR = selector.selected_[expected].mean()
        self.assertGreaterEqual(TPR, 0.99)
