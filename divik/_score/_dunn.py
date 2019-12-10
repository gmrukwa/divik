from functools import partial
from multiprocessing.pool import Pool
from typing import List, Optional
import uuid

import numpy as np
import pandas as pd

from divik._distance import DistanceMetric, make_distance
from divik._score._picker import Picker
from divik._utils import Centroids, IntLabels, Data, get_n_jobs


def dunn(data: Data, labels: IntLabels, centroids: Centroids,
         distance: DistanceMetric) -> float:
    if centroids.shape[0] == 1:
        return -np.inf
    clusters = pd.DataFrame(data).groupby(labels).apply(
        lambda cluster: cluster.values)
    intercluster = distance(centroids, centroids)
    intercluster = np.min(intercluster[intercluster != 0])
    intracluster = np.max([
        np.mean(distance(cluster, centroid.reshape(1, -1)))
        for cluster, centroid in zip(clusters, centroids)
    ])
    score = intercluster / intracluster
    return score


KMeans = 'divik.KMeans'


_DATA = {}


def _dunn(kmeans: KMeans, data: Data) -> float:
    distance = make_distance(kmeans.distance)
    if isinstance(data, str):
        data = _DATA[data]
    return dunn(data, kmeans.labels_, kmeans.cluster_centers_, distance)


class DunnPicker(Picker):
    def score(self, data: Data, estimators: List[KMeans]) -> np.ndarray:
        if self.n_jobs != 1:
            ref = str(uuid.uuid4())
            global _DATA
            _DATA[ref] = data
            score = partial(_dunn, data=ref)
            with Pool(get_n_jobs(self.n_jobs)) as pool:
                scores = pool.map(score, estimators)
            del _DATA[ref]
        else:
            scores = [_dunn(estimator, data) for estimator in estimators]
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
