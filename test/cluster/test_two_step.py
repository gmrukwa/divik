import numpy.testing as npt
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline

from divik.cluster import (
    KMeans,
    TwoStep,
)
from divik.feature_extraction import LocallyAdjustedRbfSpectralEmbedding


def data():
    return make_blobs(
        n_samples=10_000,
        n_features=2,
        centers=[[-6, 2], [5, 5], [4, -3]],
        random_state=42,
    )


def test_two_step_with_kmeans():
    X, y = data()
    kmeans = KMeans(n_clusters=3)
    ctr = TwoStep(kmeans).fit(X, y)
    assert ctr.estimator_ is not None
    assert isinstance(ctr.estimator_, KMeans)
    assert ctr.estimator_ is not kmeans
    assert ctr.estimator_.get_params() == kmeans.get_params()
    assert ctr.labels_ is not None
    assert ctr.labels_.shape == y.shape
    assert adjusted_rand_score(y, ctr.labels_) > 0.95
    y_pred = ctr.labels_
    y_pred_ = ctr.fit_predict(X, y)
    npt.assert_array_equal(y_pred, y_pred_)


@pytest.mark.filterwarnings("ignore:Graph is not fully connected")
def test_two_step_with_spectral():
    X, y = data()
    spectral = Pipeline(
        steps=[
            (
                "embedding",
                LocallyAdjustedRbfSpectralEmbedding(
                    n_components=2, random_state=42, n_neighbors=7
                ),
            ),
            ("kmeans", KMeans(n_clusters=3)),
        ]
    )
    ctr = TwoStep(spectral).fit(X, y)
    assert ctr.estimator_ is not None
    assert isinstance(ctr.estimator_, Pipeline)
    assert ctr.estimator_ is not spectral
    assert ctr.estimator_[0].get_params() == spectral[0].get_params()
    assert ctr.estimator_[1].get_params() == spectral[1].get_params()
    assert ctr.labels_ is not None
    assert ctr.labels_.shape == y.shape
    assert adjusted_rand_score(y, ctr.labels_) > 0.95
    y_pred = ctr.labels_
    y_pred_ = ctr.fit_predict(X, y)
    npt.assert_array_equal(y_pred, y_pred_)
