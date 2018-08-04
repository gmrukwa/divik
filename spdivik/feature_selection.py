"""Methods of data-driven feature selection.

feature_selection.py

Copyright 2018 Spectre Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from functools import partial
from typing import Callable, Tuple

import numpy as np

import spdivik._matlab_legacy as ml
import spdivik.types as ty

Statistic = Callable[[np.ndarray], np.ndarray]  # computes columns' scores

amplitude = partial(np.mean, axis=0)
variance = partial(np.var, axis=0)


def _allow_all(data: ty.Data, topmost: bool = True) \
        -> Tuple[ty.BoolFilter, float]:
    return np.ones((data.shape[1],), dtype=bool), -np.inf if topmost else np.inf


def select_by(data: ty.Data, statistic: Statistic, discard_up_to: int = -1,
              min_features: int = 1, min_features_percentage: float = None,
              preserve_topmost: bool = True) -> Tuple[ty.BoolFilter, float]:
    """Select features by a statistic decomposition.

    @param data: dataset to select features from. Observations in rows,
    features in columns
    @param statistic: statistic to decompose that characterizes the feature
    @param discard_up_to: number of components that will be discarded atmost.
    If the number provided is negative, all components are allowed to be
    discarded but -discard_up_to. I.e., if one wants to discard all but one
    component provides discard_up_to=-1.
    @param min_features_percentage: minimal percent of features that must be
    preserved
    @param preserve_topmost: if true, the topmost components will be preserved;
    lowest components otherwise
    @return: boolean filter preserving features and value of the threshold
    for feature filtering by the characteristic
    """
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
