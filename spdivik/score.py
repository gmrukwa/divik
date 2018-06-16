from itertools import product
from typing import Callable, List, NamedTuple, Tuple

import numpy as np
import pandas as pd

from spdivik.distance import DistanceMetric
from spdivik.types import \
    Centroids, \
    Data, \
    IntLabels, \
    Quality, \
    SegmentationMethod


Score = Callable[[Data, IntLabels, Centroids], float]


def dunn(data: Data, labels: IntLabels, centroids: Centroids,
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


ParameterValues = NamedTuple('ParameterValues', [
    ('name', str),
    ('values', List)
])


class Optimizer:
    def __init__(self, score: Score, segmentation_method: SegmentationMethod,
                 parameters: List[ParameterValues]):
        self.score = score
        self.segmentation_method = segmentation_method
        self.parameter_names, self.parameter_sets = zip(*parameters)
        assert all(len(col) > 0 for col in self.parameter_sets)

    def __call__(self, data: Data) -> Tuple[IntLabels, Centroids, Quality]:
        best_score = -np.inf
        best_result = None
        for parameter_set in product(*self.parameter_sets):
            parameters = {
                name: value for name, value
                in zip(self.parameter_names, parameter_set)
            }
            result = self.segmentation_method(data, **parameters)
            score = self.score(data, *result)
            if score > best_score:
                best_score, best_result = score, result
        return best_result + (best_score,)
