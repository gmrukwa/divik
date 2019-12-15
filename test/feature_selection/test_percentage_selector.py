import unittest

import numpy as np

import divik.feature_selection as fs


class PercentageSelectorTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.labels = np.concatenate(
            [30 * [0] + 20 * [1] + 30 * [2] + 40 * [3]])
        self.data = np.vstack(100 * [self.labels * 10.])
        self.data += np.random.randn(*self.data.shape)
        sub = self.data[:, :-40]
        sub += 5 * np.random.randn(*sub.shape)

    def test_discards_low_abundance(self):
        p = 1 - np.sum(self.labels == 0) / self.labels.size
        selector = fs.PercentageSelector('mean', p=p).fit(self.data)
        TNR = (selector.selected_[self.labels == 0] == False).mean()
        self.assertGreaterEqual(TNR, 0.99)

    def test_discards_low_variance(self):
        p = np.sum(self.labels == 3) / self.labels.size
        selector = fs.PercentageSelector('var', p=p).fit(self.data)
        TNR = (selector.selected_[self.labels == 3] == False).mean()
        self.assertGreaterEqual(TNR, 0.99)

    def test_allows_reverse(self):
        p = 1 - np.sum(self.labels == 0) / self.labels.size
        selector = fs.PercentageSelector('mean', p=p, keep_top=False
                                         ).fit(self.data)
        TPR = selector.selected_[self.labels == 0].mean()
        self.assertGreaterEqual(TPR, 0.99)
