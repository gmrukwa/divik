import unittest

import numpy as np
import numpy.testing as npt
from sklearn.datasets import make_blobs

from divik.cluster import (
    DiviK,
    GAPSearch,
    KMeans,
)


def _full_kmeans(**config):
    distance = config.get("distance", "correlation")
    normalize_rows = config.get("normalize_rows", None)
    single_kmeans = KMeans(
        n_clusters=2,
        distance=distance,
        init="percentile",
        percentile=config.get("distance_percentile", 99.0),
        max_iter=config.get("max_iter", 100),
        normalize_rows=normalize_rows,
    )
    kmeans = GAPSearch(
        single_kmeans,
        max_clusters=config.get("k_max", 50),
        n_jobs=config.get("n_jobs", None),
        seed=config.get("random_seed", 0),
        n_trials=config.get("gap_trials", 10),
        sample_size=config.get("sample_size", 10000),
        verbose=config.get("verbose", False),
    )
    return kmeans


full = _full_kmeans(distance="euclidean", max_iter=10)


class GapDivikTest(unittest.TestCase):
    def test_predict_gives_consistent_results(self):
        X, _ = make_blobs(n_samples=300, n_features=100, centers=3, random_state=42)
        model = DiviK(full, distance="euclidean")
        predictions = model.fit_predict(X)
        reproduced = model.predict(X)
        npt.assert_array_equal(predictions, reproduced)

    def test_predict_works_for_single_cluster(self):
        X, _ = make_blobs(n_samples=300, n_features=100, centers=1, random_state=42)
        model = DiviK(full, distance="euclidean")
        predictions = model.fit_predict(X)
        reproduced = model.predict(X)
        npt.assert_array_equal(predictions, reproduced)

    def test_predict_works_for_neutral_elements(self):
        X, _ = make_blobs(n_samples=300, n_features=100, centers=1, random_state=42)
        model = DiviK(full, distance="euclidean", neutral=0)
        predictions = model.fit_predict(X)
        reproduced = model.predict(X)
        npt.assert_array_equal(predictions, reproduced)

    def test_produces_hierarchy_for_numerous_clusters(self):
        X, _ = make_blobs(n_samples=2000, n_features=100, centers=20, random_state=42)
        model = DiviK(full, distance="euclidean")
        model.fit(X)
        assert any(sub is not None for sub in model.result_.subregions)

    def test_predict_works_for_numerous_clusters(self):
        X, _ = make_blobs(n_samples=600, n_features=100, centers=20, random_state=42)
        model = DiviK(full, distance="euclidean")
        predictions = model.fit_predict(X)
        reproduced = model.predict(X)
        npt.assert_array_equal(predictions, reproduced)

    def test_works_with_pool(self):
        X, _ = make_blobs(n_samples=600, n_features=100, centers=20, random_state=42)
        sequential = DiviK(full, distance="euclidean")
        parallel = DiviK(full, distance="euclidean", n_jobs=-1)
        expected = sequential.fit_predict(X)
        labels1 = parallel.fit_predict(X)
        labels2 = parallel.predict(X)
        npt.assert_array_equal(expected, labels1)
        npt.assert_array_equal(expected, labels2)

    def test_works_with_pool_and_sampling(self):
        X, _ = make_blobs(n_samples=1000, n_features=100, centers=5, random_state=43)
        sequential = DiviK(full, distance="euclidean", rejection_size=5)
        parallel = DiviK(full, distance="euclidean", n_jobs=-1, rejection_size=5)
        expected = sequential.fit_predict(X)
        labels1 = parallel.fit_predict(X)
        labels2 = parallel.predict(X)
        npt.assert_array_equal(expected, labels1)
        npt.assert_array_equal(expected, labels2)

    def test_transforms_to_n_clusters_dimensions(self):
        X, _ = make_blobs(n_samples=600, n_features=100, centers=20, random_state=42)
        model = DiviK(full, distance="euclidean")
        X_trans = model.fit_transform(X)
        assert X_trans.shape[1] == model.n_clusters_

    def test_is_closest_to_predicted(self):
        X, _ = make_blobs(n_samples=400, n_features=100, centers=20, random_state=42)
        # This is true only with full feature's space actually, since otherwise
        # dimensionality may differ too much and influence the distance
        # comparison.
        model = DiviK(
            full,
            distance="euclidean",
            minimal_features_percentage=1.0,
            features_percentage=1.0,
        )
        y_pred = model.fit_predict(X)
        X_trans = model.transform(X)
        closests = [np.argmin(row) for row in X_trans]
        npt.assert_array_equal(y_pred, closests)
