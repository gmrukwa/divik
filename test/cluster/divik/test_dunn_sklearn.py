import unittest
import numpy as np
import numpy.testing as npt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from divik.cluster._divik._sklearn import DiviK
from divik._cli.dunn_divik import _fast_kmeans, _full_kmeans


fast = _fast_kmeans(distance='euclidean')
full = _full_kmeans(distance='euclidean')


class DunnDivikTest(unittest.TestCase):
    def test_predict_gives_consistent_results(self):
        X, _ = make_blobs(n_samples=100, n_features=100, centers=3,
                          random_state=42)
        model = DiviK(full, fast, distance='euclidean')
        predictions = model.fit_predict(X)
        reproduced = model.predict(X)
        npt.assert_array_equal(predictions, reproduced)

    def test_predict_works_for_single_cluster(self):
        X, _ = make_blobs(n_samples=100, n_features=100, centers=1,
                          random_state=42)
        model = DiviK(full, fast, distance='euclidean')
        predictions = model.fit_predict(X)
        reproduced = model.predict(X)
        npt.assert_array_equal(predictions, reproduced)

    def test_produces_hierarchy_for_numerous_clusters(self):
        X, _ = make_blobs(n_samples=200, n_features=100, centers=20,
                          random_state=42)
        model = DiviK(full, fast, distance='euclidean')
        model.fit(X)
        self.assertTrue(any(
            sub is not None for sub in model.result_.subregions))

    def test_predict_works_for_numerous_clusters(self):
        X, _ = make_blobs(n_samples=200, n_features=100, centers=20,
                          random_state=42)
        model = DiviK(full, fast, distance='euclidean')
        predictions = model.fit_predict(X)
        reproduced = model.predict(X)
        npt.assert_array_equal(predictions, reproduced)

    def test_works_with_pool(self):
        X, _ = make_blobs(n_samples=200, n_features=100, centers=20,
                          random_state=42)
        sequential = DiviK(full, fast, distance='euclidean')
        parallel = DiviK(full, fast, distance='euclidean', n_jobs=-1)
        expected = sequential.fit_predict(X)
        labels1 = parallel.fit_predict(X)
        labels2 = parallel.predict(X)
        npt.assert_array_equal(expected, labels1)
        npt.assert_array_equal(expected, labels2)

    def test_works_with_pool_and_sampling(self):
        X, _ = make_blobs(n_samples=1000, n_features=100, centers=5,
                          random_state=42)
        sequential = DiviK(full, fast, distance='euclidean',
                           rejection_size=5)
        parallel = DiviK(full, fast, distance='euclidean',
                         n_jobs=-1, rejection_size=5)
        expected = sequential.fit_predict(X)
        labels1 = parallel.fit_predict(X)
        labels2 = parallel.predict(X)
        npt.assert_array_equal(expected, labels1)
        npt.assert_array_equal(expected, labels2)

    def test_has_decent_performance_on_numerous_clusters(self):
        X, y = make_blobs(n_samples=200, n_features=100, centers=20,
                          random_state=42)
        model = DiviK(full, fast, distance='euclidean')
        y_pred = model.fit_predict(X)
        score = adjusted_rand_score(y, y_pred)
        self.assertGreaterEqual(score, 0.95)

    def test_transforms_to_n_clusters_dimensions(self):
        X, _ = make_blobs(n_samples=200, n_features=100, centers=20,
                          random_state=42)
        model = DiviK(full, fast, distance='euclidean')
        X_trans = model.fit_transform(X)
        self.assertEqual(X_trans.shape[1], model.n_clusters_)

    def test_is_closest_to_predicted(self):
        X, _ = make_blobs(n_samples=200, n_features=100, centers=20,
                          random_state=42)
        # This is true only with full feature's space actually, since otherwise
        # dimensionality may differ too much and influence the distance
        # comparison.
        model = DiviK(full, fast, distance='euclidean',
                      minimal_features_percentage=1.0,
                      features_percentage=1.0)
        y_pred = model.fit_predict(X)
        X_trans = model.transform(X)
        closests = [np.argmin(row) for row in X_trans]
        npt.assert_array_equal(y_pred, closests)
