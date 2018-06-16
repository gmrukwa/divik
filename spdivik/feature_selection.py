from functools import partial
from typing import Callable, Tuple

import numpy as np

import spdivik._matlab_legacy as ml
import spdivik.types as ty

Statistic = Callable[[np.ndarray], np.ndarray]  # computes columns' scores

amplitude = partial(np.mean, axis=0)
variance = partial(np.var, axis=0)


def _allow_all(data: ty.Data, topmost: bool=True) -> Tuple[ty.BoolFilter, float]:
    return np.ones((data.shape[1],), dtype=bool), -np.inf if topmost else np.inf


def select_by(data: ty.Data, statistic: Statistic, discard_up_to: int = -1,
              min_features: int = 1, min_features_percentage: float = None,
              preserve_topmost: bool=True) -> Tuple[ty.BoolFilter, float]:
    if min_features_percentage is not None:
        min_features = int(data.shape[1] * min_features_percentage + .5)
    scores = statistic(data)
    if not preserve_topmost:
        scores = -scores
    thresholds = ml.find_thresholds(scores, throw_on_engine_error=False)
    desired_thresholds = thresholds[:discard_up_to]
    for threshold in reversed(desired_thresholds):
        selected = scores >= threshold
        if selected.sum() >= min_features:
            return selected, (threshold if preserve_topmost else -threshold)
    return _allow_all(data, preserve_topmost)
