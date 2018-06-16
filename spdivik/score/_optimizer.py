from itertools import product

import numpy as np
from typing import Callable, NamedTuple, List, Tuple

from spdivik.types import Data, IntLabels, Centroids, SegmentationMethod, \
    Quality

Score = Callable[[Data, IntLabels, Centroids], float]
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