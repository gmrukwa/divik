from functools import partial
from multiprocessing.pool import Pool
from typing import List, Optional

import numpy as np
import pandas as pd

from divik._distance import DistanceMetric, make_distance
from divik._score._picker import Picker
from divik._utils import Centroids, IntLabels, Data


def dunn(data: Data, labels: IntLabels, centroids: Centroids,
         distance: DistanceMetric) -> float:
    if centroids.shape[0] == 1:
        return -np.inf
    clusters = pd.DataFrame(data).groupby(labels).apply(lambda cluster: cluster.values)
    intercluster = distance(centroids, centroids)
    intercluster = np.min(intercluster[intercluster != 0])
    intracluster = np.max([
        np.mean(distance(cluster, centroid.reshape(1, -1)))
        for cluster, centroid in zip(clusters, centroids)
    ])
    score = intercluster / intracluster
    return score


KMeans = 'divik.KMeans'


def _dunn(kmeans: KMeans, data: Data) -> float:
    distance = make_distance(kmeans.distance)
    return dunn(data, kmeans.labels_, kmeans.cluster_centers_, distance)


class DunnPicker(Picker):
    def score(self, data: Data, estimators: List[KMeans], pool: Pool=None) \
            -> np.ndarray:
        score = partial(_dunn, data=data)
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
