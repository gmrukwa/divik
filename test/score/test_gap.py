import unittest

import numpy as np
import pandas as pd

import divik.cluster as km
import divik._score as sc


def dummy_kmeans(labels, cluster_centers):
    kmeans = km.KMeans(n_clusters=len(cluster_centers), distance='euclidean',
                       init='extreme', max_iter=10)
    kmeans.labels_ = labels
    kmeans.cluster_centers_ = cluster_centers
    return kmeans


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
        # make labels
        labels = np.zeros(shape=(300,), dtype=int)
        labels[idx] = 1
        centroids = pd.DataFrame(self.data).groupby(labels).mean().values
        # generate worse labeling
        idx = np.random.randint(0, 300, 50)
        worse_labels = labels[:]
        worse_labels[idx] = 1 - worse_labels[idx]
        worse_centroids = pd.DataFrame(
            self.data).groupby(worse_labels).mean().values
        self.worse = dummy_kmeans(worse_labels, worse_centroids)
        self.normal = dummy_kmeans(labels, centroids)

    def test_computes_score(self):
        score = sc.gap(self.data, self.normal)
        self.assertFalse(np.isnan(score))

    def test_good_labeling_has_greater_score(self):
        better = sc.gap(self.data, self.normal)
        worse = sc.gap(self.data, self.worse)
        self.assertGreater(better, worse)

    def test_returns_std_if_requested(self):
        gap, std = sc.gap(self.data, self.normal, return_deviation=True)
        self.assertIsNotNone(gap)
        self.assertIsNotNone(std)
