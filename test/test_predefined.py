import unittest
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import divik.distance as dst
import divik.predefined as pre

N_OBSERVATIONS = 1000
N_FEATURES = 40
N_VARYING = 10
N_UPREGULATED = 20
N_SEPARATED = 100


class DataBoundTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.data = np.random.randn(N_OBSERVATIONS, N_FEATURES)
        # vary some features
        idx = np.random.randint(0, N_FEATURES, N_VARYING)
        cls.data[:, tuple(idx)] *= 3
        # upregulate some features
        idx = np.concatenate((
            np.random.randint(0, N_FEATURES, N_UPREGULATED - N_VARYING),
            idx))
        cls.data[:, tuple(idx)] += 100
        # separate a group
        idx = np.random.randint(0, N_OBSERVATIONS, N_SEPARATED)
        cls.data[tuple(idx), :] += 500
        # normalize data
        cls.data = cls.data - np.mean(cls.data, axis=0)
        cls.data = cls.data / np.max(cls.data, axis=0)


pool = Pool()


class TestBasic(DataBoundTestCase):
    @classmethod
    def setUpClass(cls):
        super(TestBasic, cls).setUpClass()
        progress_bar = tqdm(desc='divik', total=cls.data.shape[0])
        distance = dst.KnownMetric.euclidean.value
        divik = pre.basic(gap_trials=10, pool=pool,
                          correction_of_gap=False,
                          progress_reporter=progress_bar,
                          distance=distance,
                          use_logfilters=True)
        cls.result = divik(cls.data)

    def test_constructs_runnable_pipeline(self):
        self.assertIsNotNone(self.result)

    def test_splits_test_data_into_two_topmost_groups(self):
        self.assertEqual(len(self.result.subregions), 2)

    def test_makes_only_topmost_split(self):
        self.assertIsNone(self.result.subregions[0])
        self.assertIsNone(self.result.subregions[1])

    def test_scores_segmentation(self):
        self.assertFalse(np.isnan(self.result.quality))
