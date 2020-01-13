import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

from divik.core import Data


KMeans = 'divik.KMeans'


def dunn(kmeans: KMeans, data: Data) -> float:
    """Compute Dunn's index for the clustering

    Parameters
    ----------
    kmeans : KMeans
        KMeans object fitted to the data

    data : array, shape (n_samples, n_features)
        Clustered data

    Returns
    -------
    dunn_index : float
        Value of Dunn's index for the clustering of data
    """
    if kmeans.cluster_centers_.shape[0] == 1:
        return -np.inf
    clusters = pd.DataFrame(data).groupby(kmeans.labels_).apply(np.asarray)
    intercluster = dist.pdist(kmeans.cluster_centers_, kmeans.distance)
    intercluster = np.min(intercluster[intercluster != 0])
    intracluster = np.max([
        np.mean(dist.cdist(cluster, centroid.reshape(1, -1), kmeans.distance))
        for cluster, centroid in zip(clusters, kmeans.cluster_centers_)
    ])
    score = intercluster / intracluster
    return score
