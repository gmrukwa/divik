import logging
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

from divik.core import (
    Data,
    configurable,
    maybe_pool,
)
from divik.sampler import BaseSampler, StratifiedSampler

KMeans = "divik.cluster.KMeans"
_BIG_PRIME = 49277


def _inter_centroid(kmeans: KMeans, data: Data, labels=None):
    d = dist.pdist(kmeans.cluster_centers_, kmeans.distance)
    return np.min(d[d != 0])


def _inter_closest(kmeans: KMeans, data: Data, labels=None):
    if labels is None:
        labels = kmeans.labels_
    d = np.inf
    for label in np.arange(kmeans.n_clusters - 1):
        grp = label == labels
        non_grp = label < labels
        dst = dist.cdist(data[grp], data[non_grp], metric=kmeans.distance)
        d = np.minimum(d, dst.min())
    return d


def _intra_avg(kmeans: KMeans, data: Data, labels=None):
    if labels is None:
        labels = kmeans.labels_
    clusters = pd.DataFrame(data).groupby(labels).apply(np.asarray)
    return np.max(
        [
            np.mean(dist.cdist(cluster, centroid.reshape(1, -1), kmeans.distance))
            for cluster, centroid in zip(clusters, kmeans.cluster_centers_)
        ]
    )


def _intra_furthest(kmeans: KMeans, data: Data, labels=None):
    def max_distance(group):
        group = np.asarray(group)
        d = dist.pdist(group, metric=kmeans.distance)
        # 0 is intracluster distance for cluster with one observation
        return np.max(d, initial=0.0)

    if labels is None:
        labels = kmeans.labels_
    return pd.DataFrame(data).groupby(labels).apply(max_distance).max()


_INTER = {
    "centroid": _inter_centroid,
    "closest": _inter_closest,
}
_INTRA = {
    "avg": _intra_avg,
    "furthest": _intra_furthest,
}


@configurable
def dunn(kmeans: KMeans, data: Data, inter="centroid", intra="avg") -> float:
    """Compute Dunn's index for the clustering

    Parameters
    ----------
    kmeans : KMeans
        KMeans object fitted to the data

    data : array, shape (n_samples, n_features)
        Clustered data

    inter : {'centroid', 'closest'}
        Method of computing intercluster distance
        - centroid - uses distances between centroids
        - closest - uses distance between closest members of separate clusters

    intra : {'avg', 'furthest}
        Method of computing intracluster distance
        - avg - uses average distance to the centroid
        - furthest - uses distance between the furthest cluster members

    Returns
    -------
    dunn_index : float
        Value of Dunn's index for the clustering of data
    """
    if kmeans.cluster_centers_.shape[0] == 1:
        return -np.inf
    if inter not in _INTER:
        msg = (
            f"Unsupported intercluster distance {inter}. "
            + f"Supported: {list(_INTER.keys())}"
        )
        logging.error(msg)
        raise ValueError(msg)
    if intra not in _INTRA:
        msg = (
            f"Unsupported intracluster distance {intra}. "
            + f"Supported: {list(_INTRA.keys())}"
        )
        logging.error(msg)
        raise ValueError(msg)
    intercluster = _INTER[inter](kmeans, data)
    intracluster = _INTRA[intra](kmeans, data)
    score = intercluster / intracluster
    return score


def _sample_distances(
    seed: int, sampler: BaseSampler, kmeans: KMeans, inter="centroid", intra="avg"
):
    data = sampler.get_sample(seed)
    labels = kmeans.predict(data)
    inter_ = _INTER[inter](kmeans, data, labels)
    intra_ = _INTRA[intra](kmeans, data, labels)
    return inter_, intra_


@configurable
def sampled_dunn(
    kmeans: KMeans,
    data: Data,
    sample_size: Union[int, float] = 1000,
    n_jobs: int = None,
    seed: int = 0,
    n_trials: int = 10,
    inter="closest",
    intra="furthest",
) -> float:
    data_ = StratifiedSampler(n_rows=sample_size, n_samples=n_trials).fit(
        data, kmeans.labels_
    )
    seeds = list(seed + np.arange(n_trials) * _BIG_PRIME)
    with data_.parallel() as d, maybe_pool(
        n_jobs, initializer=d.initializer, initargs=d.initargs
    ) as pool:
        distances = partial(
            _sample_distances, sampler=d, kmeans=kmeans, inter=inter, intra=intra
        )
        inter_, intra_ = np.array(pool.map(distances, seeds)).T
    s_inter = inter_.std()
    s_intra = intra_.std()
    return (inter_.min() - s_inter) / (intra_.max() + s_intra)
