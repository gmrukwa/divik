import unittest

import numpy as np

import divik.feature_selection as fs


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


class OutlierAbundanceAndVarianceSelectorTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.labels = np.concatenate(
            [6 * [0] + 5 * [1] + 25 * [2] + 50 * [3] + 3 * [4] + 4 * [5]])
        self.data = np.vstack(1000 * [self.labels * 10.])
        self.data += np.random.randn(*self.data.shape)
        self.outlier_mean = np.logical_or(self.labels == 4, self.labels == 5)
        self.high_var = np.zeros_like(self.outlier_mean)
        self.high_var[np.where(self.labels == 2)[0][:5]] = True
        shape = self.data[:, self.high_var].shape
        self.data[:, self.high_var] += 5 * np.random.randn(*shape)

    def test_discards_outlier_abundance(self):
        selector = fs.OutlierAbundanceAndVarianceSelector().fit(self.data)
        TNR = (selector.selected_[self.outlier_mean] == False).mean()
        self.assertGreaterEqual(TNR, 0.95)

    def test_discards_outlier_variance(self):
        selector = fs.OutlierAbundanceAndVarianceSelector().fit(self.data)
        TNR = (selector.selected_[self.high_var] == False).mean()
        self.assertGreaterEqual(TNR, 0.95)

    def test_selects_percentage_of_features(self):
        N = self.labels.size
        p = 5. / N
        selector = fs.OutlierAbundanceAndVarianceSelector(p=p).fit(self.data)
        self.assertAlmostEqual(selector.selected_.mean(), p, places=2)
        p = 20. / N
        selector = fs.OutlierAbundanceAndVarianceSelector(p=p).fit(self.data)
        self.assertAlmostEqual(selector.selected_.mean(), p, places=2)
