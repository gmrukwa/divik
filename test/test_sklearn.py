import unittest
import numpy.testing as npt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

import divik.distance as dst

from divik._sklearn import DiviK


class DivikTest(unittest.TestCase):
    def test_predict_gives_consistent_results(self):
        X, _ = make_blobs(n_samples=100, n_features=100, centers=3,
                          random_state=42)
        model = DiviK(distance=dst.KnownMetric.euclidean.value)
        predictions = model.fit_predict(X)
        reproduced = model.predict(X)
        npt.assert_array_equal(predictions, reproduced)

    def test_predict_works_for_single_cluster(self):
        X, _ = make_blobs(n_samples=100, n_features=100, centers=1,
                          random_state=42)
        model = DiviK(distance=dst.KnownMetric.euclidean.value)
        predictions = model.fit_predict(X)
        reproduced = model.predict(X)
        npt.assert_array_equal(predictions, reproduced)

    def test_produces_hierarchy_for_numerous_clusters(self):
        X, _ = make_blobs(n_samples=200, n_features=100, centers=20,
                          random_state=42)
        model = DiviK(distance=dst.KnownMetric.euclidean.value)
        model.fit(X)
        self.assertTrue(any(
            sub is not None for sub in model.result_.subregions))

    def test_predict_works_for_numerous_clusters(self):
        X, _ = make_blobs(n_samples=200, n_features=100, centers=20,
                          random_state=42)
        model = DiviK(distance=dst.KnownMetric.euclidean.value)
        predictions = model.fit_predict(X)
        reproduced = model.predict(X)
        npt.assert_array_equal(predictions, reproduced)

    def test_works_with_pool(self):
        X, _ = make_blobs(n_samples=200, n_features=100, centers=20,
                          random_state=42)
        sequential = DiviK(distance=dst.KnownMetric.euclidean.value)
        parallel = DiviK(distance=dst.KnownMetric.euclidean.value, n_jobs=-1)
        expected = sequential.fit_predict(X)
        labels = parallel.fit_predict(X)
        npt.assert_array_equal(expected, labels)

    def test_has_decent_performance_on_numerous_clusters(self):
        X, y = make_blobs(n_samples=200, n_features=100, centers=20,
                          random_state=42)
        model = DiviK(distance=dst.KnownMetric.euclidean.value, n_jobs=-1)
        y_pred = model.fit_predict(X)
        score = adjusted_rand_score(y, y_pred)
        self.assertGreaterEqual(score, 0.95)
