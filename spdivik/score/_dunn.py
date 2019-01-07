from functools import partial
from multiprocessing.pool import Pool
from typing import List, Optional

import numpy as np
import pandas as pd

from spdivik.kmeans._core import KMeans, parse_distance
from spdivik.distance import DistanceMetric
from spdivik.score._picker import Picker
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


def dunn(kmeans: KMeans, data: Data) -> float:
    distance = parse_distance(kmeans.distance)
    return dunn_(data, kmeans.labels_, kmeans.cluster_centers_, distance)


class DunnPicker(Picker):
    def score(self, data: Data, estimators: List[KMeans], pool: Pool=None) \
            -> np.ndarray:
        score = partial(dunn, data=data)
        if pool:
            scores = pool.map(score, estimators)
        else:
            scores = [score(estimator) for estimator in estimators]
        return np.array(scores)

    def select(self, scores: np.ndarray) -> Optional[int]:
        return int(np.argmax(scores))

    def report(self, estimators: List[KMeans], scores: np.ndarray) \
            -> pd.DataFrame:
        best = self.select(scores)
        suggested = np.zeros((scores.size,), dtype=bool)
        suggested[best] = True
        return pd.DataFrame(
            data={
                'number_of_clusters': [e.n_clusters for e in estimators],
                'Dunn': scores.ravel(),
                'suggested_number_of_clusters': suggested
            }, columns=['number_of_clusters', 'Dunn'])
