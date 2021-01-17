from typing import NamedTuple

import numpy as np
from kneed import KneeLocator

FeaturesSelection = NamedTuple(
    "FeaturesSelection",
    [
        ("score", np.ndarray),
        ("index", np.ndarray),
        ("selection", np.ndarray),
        ("threshold", float),
    ],
)


def _gradient(values: np.ndarray, order: int = 1) -> np.ndarray:
    result = values
    for _ in range(order):
        result = np.gradient(result)
    return result


def _plateau_point(sorted_scores) -> int:
    gradient = np.abs(_gradient(sorted_scores))
    is_plateau = gradient <= np.percentile(gradient, 1)
    plateau_points = np.nonzero(is_plateau)[0]
    return int(np.median(plateau_points))


def _knee_point(decreasing_segment: np.ndarray) -> int:
    locator = KneeLocator(
        x=np.arange(decreasing_segment.size, dtype=int),
        y=decreasing_segment,
        S=1.0,
        curve="convex",
        direction="decreasing",
    )
    assert locator.knee is not None
    return locator.knee


def select_features(feature_scores) -> FeaturesSelection:
    feature_scores = feature_scores.ravel()
    index = np.argsort(-feature_scores)
    score = feature_scores[index]
    plateau_point = _plateau_point(score)
    concave_up_segment = score[:plateau_point]
    knee_location = _knee_point(concave_up_segment)
    threshold = score[knee_location]
    selection = feature_scores >= threshold
    return FeaturesSelection(score, index, selection, threshold)
