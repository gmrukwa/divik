import numpy as np
import pandas as pd

from spdivik.distance import DistanceMetric
from spdivik.types import \
    Centroids, \
    Data, \
    IntLabels



def dunn(data: Data, labels: IntLabels, centroids: Centroids,
         distance: DistanceMetric) -> float:
    clusters = pd.DataFrame(data).groupby(labels).apply(lambda cluster: cluster.values)
    intercluster = distance(centroids, centroids)
    intercluster = np.min(intercluster[intercluster != 0])
    intracluster = np.max([
        np.mean(distance(cluster, centroid.reshape(1, -1)))
        for cluster, centroid in zip(clusters, centroids)
    ])
    score = intercluster / intracluster
    return score
