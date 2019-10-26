from functools import partial
from itertools import product
from multiprocessing import Pool

import numpy as np
from typing import Callable, NamedTuple, List, Tuple

from divik._utils import Centroids, IntLabels, Data, Quality, SegmentationMethod

Score = Callable[[Data, IntLabels, Centroids], float]
ParameterValues = NamedTuple('ParameterValues', [
    ('name', str),
    ('values', List)
])
ClusteringResult = Tuple[IntLabels, Centroids, Quality]


def _split(data, *args, score: Score, split: SegmentationMethod,
           parameter_names: List[str]) -> ClusteringResult:
    parameters = {name: value for name, value in zip(parameter_names, args)}
    labels, centroids = split(data, **parameters)
    quality = score(data, labels, centroids)
    return labels, centroids, quality


class Optimizer:
    def __init__(self, score: Score, segmentation_method: SegmentationMethod,
                 parameters: List[ParameterValues], pool: Pool=None):
        parameter_names, self.parameter_sets = zip(*parameters)
        assert all(len(col) > 0 for col in self.parameter_sets)
        self.pool = pool
        self.split = partial(_split, score=score, split=segmentation_method,
                             parameter_names=[name for name in parameter_names])

    def __call__(self, data: Data) -> ClusteringResult:
        split = partial(self.split, data)
        if self.pool is not None:
            results = self.pool.starmap(split, product(*self.parameter_sets))
        else:
            results = [split(*args) for args in product(*self.parameter_sets)]
        idx_of_best = np.argmax([quality for _, _, quality in results])
        return results[idx_of_best]
