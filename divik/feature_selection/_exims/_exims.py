from functools import partial
from multiprocessing import Pool
from typing import Callable, Tuple

import numpy as np
from tqdm import tqdm

from divik.feature_selection._exims._structness import structness


class pipe:
    def __init__(self, *functions):
        self.functions = functions

    def __call__(self, *args, **kwargs):
        result = self.functions[0](*args, **kwargs)
        for func in self.functions[1:]:
            result = func(result)
        return result


def progress_bar(description: str = None):
    return partial(tqdm, desc=description)


def pmap(func, collection, **kwargs):
    with Pool() as pool:
        return pool.map(func, collection, **kwargs)


def apply(func, collection):
    return [func(element) for element in collection]


def for_each(func, lazy: bool = True, parallel: bool = False, **kwargs):
    if parallel:
        return partial(pmap, func, **kwargs)
    if lazy:
        return partial(map, func)
    else:
        return partial(apply, func)


def as_image(data: np.ndarray, x: np.ndarray, y: np.ndarray, default=-1) -> np.ndarray:
    x, y = x.astype(int), y.astype(int)
    translated_x, translated_y = x - np.min(x), y - np.min(y)
    rows, columns = int(np.max(translated_y) + 1), int(np.max(translated_x) + 1)
    if len(data.shape) < 2:
        data = data.reshape((data.shape[0], 1))
    cube = default * np.ones((rows, columns, data.shape[1]))
    cube[translated_y, translated_x] = data
    return cube


_IGNORED = -1
_Feature = np.ndarray
_Structness = float
_FeatureProcessor = Callable[[_Feature], Tuple[_Structness, _Structness]]
_remove_channel_dimension = partial(np.squeeze, axis=2)


def _feature_processor(x: np.ndarray, y: np.ndarray) -> _FeatureProcessor:
    # noinspection PyTypeChecker
    return pipe(
        partial(as_image, x=x, y=y, default=_IGNORED),
        _remove_channel_dimension,
        partial(structness, ignored=[_IGNORED]),
    )


def _normalize_columns(matrix) -> np.ndarray:
    matrix = np.array(matrix, dtype=float)
    matrix += np.finfo(float).eps
    assert len(matrix.shape) == 2
    return matrix / np.max(matrix, axis=0)


_as_features = np.transpose
_normalize_structness_by_kind = _normalize_columns
_sumarize_structness_by_feature = pipe(partial(np.sum, axis=1), np.ravel)
FeaturesStructness = np.ndarray
_StructnessEstimator = Callable[[np.ndarray], FeaturesStructness]


def _estimator(structness_: _FeatureProcessor) -> _StructnessEstimator:
    # noinspection PyTypeChecker
    return pipe(
        _as_features,
        progress_bar("feature structness"),
        for_each(structness_, parallel=True),
        _normalize_structness_by_kind,
        _sumarize_structness_by_feature,
    )


def exims(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> FeaturesStructness:
    structness_estimator = _estimator(_feature_processor(x, y))
    return structness_estimator(data)
