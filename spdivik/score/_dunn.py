import numpy as np
import pandas as pd

from spdivik.kmeans._core import KMeans, parse_distance
from spdivik.distance import DistanceMetric
from spdivik.types import Data, IntLabels, Centroids


def dunn_(data: Data, labels: IntLabels, centroids: Centroids,
          distance: DistanceMetric) -> float:
    if centroids.shape[0] == 1:
        raise ValueError('At least 2 clusters are required.')
    clusters = pd.DataFrame(data).groupby(labels).apply(lambda cluster: cluster.values)
    intercluster = distance(centroids, centroids)
    intercluster = np.min(intercluster[intercluster != 0])
    intracluster = np.max([
        np.mean(distance(cluster, centroid.reshape(1, -1)))
        for cluster, centroid in zip(clusters, centroids)
    ])
    score = intercluster / intracluster
    return score


def dunn(data: Data, kmeans: KMeans) -> float:
    distance = parse_distance(kmeans.distance)
    return dunn_(data, kmeans.labels_, kmeans.cluster_centers_, distance)
