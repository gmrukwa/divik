import unittest

import numpy as np

import divik.feature_selection._outlier as fs


class HubertaTest(unittest.TestCase):
    def test_huberta(self):
        data = np.array([7.0, 6.4, 8.7, 5.8, 6.3, 6.4, 4.6, 6.1, 16.3])
        outliers = fs.huberta_outliers(data)
        self.assertEqual(outliers.sum(), 3)
        self.assertEqual(data[outliers][0], 5.8)
        self.assertEqual(data[outliers][1], 4.6)
        self.assertEqual(data[outliers][2], 16.3)


if __name__ == '__main__':
    unittest.main()
