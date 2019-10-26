import unittest

from functools import partial

import numpy as np
import pandas as pd

import divik._distance as dst
import divik._kmeans as km
import divik._score as sc


class TestGap(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.data = np.random.randn(300, 100)
        # vary some features
        idx = np.random.randint(0, 100, 20)
        self.data[:, tuple(idx)] *= 3
        # upregulate some features
        idx = np.random.randint(0, 100, 80)
        self.data[:, tuple(idx)] += 100
        # separate a group
        idx = np.random.randint(0, 300, 100)
        self.data[tuple(idx), :] += 100
        self.labels = np.zeros(shape=(300,), dtype=int)
        self.labels[idx] = 1
        self.centroids = pd.DataFrame(self.data).groupby(
            self.labels).mean().values
        # generate worse labeling
        idx = np.random.randint(0, 300, 50)
        self.worse_labels = self.labels[:]
        self.worse_labels[idx] = 1 - self.worse_labels[idx]
        self.worse_centroids = pd.DataFrame(self.data).groupby(
            self.worse_labels).mean().values
        self.distance = dst.ScipyDistance(dst.KnownMetric.euclidean)
        kmeans = km._KMeans(km.Labeling(self.distance),
                            km.ExtremeInitialization(self.distance),
                            number_of_iterations=10)
        self.split = partial(kmeans, number_of_clusters=2)
        self.gap = partial(sc.gap, distance=self.distance, split=self.split)

    def test_computes_score(self):
        score = self.gap(self.data, self.labels, self.centroids)
        self.assertFalse(np.isnan(score))

    def test_good_labeling_has_greater_score(self):
        better = self.gap(self.data, self.labels, self.centroids)
        worse = self.gap(self.data, self.worse_labels, self.worse_centroids)
        self.assertGreater(better, worse)

    def test_returns_std_if_requested(self):
        gap, std = self.gap(self.data, self.labels, self.centroids,
                            return_deviation=True)
        self.assertIsNotNone(gap)
        self.assertIsNotNone(std)
