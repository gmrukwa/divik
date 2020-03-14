import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

from divik.core import configurable, Data


KMeans = 'divik.cluster.KMeans'


def _inter_centroid(kmeans: KMeans, data: Data):
    d = dist.pdist(kmeans.cluster_centers_, kmeans.distance)
    return np.min(d[d != 0])


def _inter_closest(kmeans: KMeans, data: Data):
    d = np.inf
    for label in np.arange(kmeans.n_clusters - 1):
        grp = label == kmeans.labels_
        non_grp = label < kmeans.labels_
        dst = dist.cdist(data[grp], data[non_grp], metric=kmeans.distance)
        d = np.minimum(d, dst.min())
    return d


def _intra_avg(kmeans: KMeans, data: Data):
    clusters = pd.DataFrame(data).groupby(kmeans.labels_).apply(np.asarray)
    return np.max([
        np.mean(dist.cdist(cluster, centroid.reshape(1, -1), kmeans.distance))
        for cluster, centroid in zip(clusters, kmeans.cluster_centers_)
    ])


def _intra_furthest(kmeans: KMeans, data: Data):
    def max_distance(group):
        group = np.asarray(group)
        d = dist.pdist(group, metric=kmeans.distance)
        return np.max(d)
    return pd.DataFrame(data).groupby(kmeans.labels_).apply(max_distance).max()


_INTER = {
    'centroid': _inter_centroid,
    'closest': _inter_closest,
}
_INTRA = {
    'avg': _intra_avg,
    'furthest': _intra_furthest,
}


@configurable
def dunn(kmeans: KMeans, data: Data, inter='centroid', intra='avg') -> float:
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
        raise ValueError(f'Unsupported intercluster distance {inter}. '
                         f'Supported: {list(_INTER.keys())}')
    if intra not in _INTRA:
        raise ValueError(f'Unsupported intracluster distance {intra}. '
                         f'Supported: {list(_INTRA.keys())}')
    intercluster = _INTER[inter](kmeans, data)
    intracluster = _INTRA[intra](kmeans, data)
    score = intercluster / intracluster
    return score
