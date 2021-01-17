import unittest

import numpy as np

import divik.feature_selection._outlier as fs


class HubertaTest(unittest.TestCase):
    def test_huberta(self):
        data = np.array([7.0, 6.4, 8.7, 5.8, 6.3, 6.4, 4.6, 6.1, 16.3])
        outliers = fs.huberta_outliers(data)
        assert outliers.sum() == 3
        assert data[outliers][0] == 5.8
        assert data[outliers][1] == 4.6
        assert data[outliers][2] == 16.3


if __name__ == "__main__":
    unittest.main()
