from functools import partial
from typing import (
    Callable,
    List,
    Tuple,
)

import numpy as np
from skimage import feature as ft

from ._matlab_alike import n_quantiles

_DISCRETIZATION_LEVELS = 8
_quantile_thresholds = partial(n_quantiles, N=_DISCRETIZATION_LEVELS - 1)
_greycomatrix_backend = partial(
    ft.greycomatrix,
    distances=[1, np.sqrt(2), 1, np.sqrt(2)],
    angles=np.radians([0.0, -45.0, -90.0, -135.0]),
    symmetric=False,
    normed=False,
)


def _greycomatrix(discrete_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    all_allowed = np.all(mask)
    if all_allowed:
        levels = _DISCRETIZATION_LEVELS
    else:
        levels = _DISCRETIZATION_LEVELS + 1
        discrete_image[~mask] = np.max(discrete_image[mask]) + 1
    gcm = _greycomatrix_backend(discrete_image, levels=levels)
    if not all_allowed:
        gcm = gcm[:-1, :-1, :, :]
    return np.dstack([gcm[:, :, i, i] for i in range(gcm.shape[2])])


def _ignorance_mask(image: np.ndarray, ignored: List) -> np.ndarray:
    mask = np.ones(image.shape, dtype=bool)
    for value in ignored:
        mask = np.logical_and(mask, image != value)
    return mask


def _discretize(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = image.astype(float)
    image /= np.max(image[mask])
    thresholds = np.hstack([_quantile_thresholds(image[mask].ravel()), (1.0,)])
    discrete = np.zeros(image.shape, dtype=np.uint8)
    for i in range(len(thresholds) - 1, -1, -1):
        pixels_of_interest = np.logical_and(image <= thresholds[i], mask)
        discrete[pixels_of_interest] = i
    return discrete


_split_directions = partial(np.rollaxis, axis=2)
_EXPECTED_BLOCK_SHAPE = 2 * (_DISCRETIZATION_LEVELS / 2,)


def _block_structness(greycomatrix_block: np.ndarray) -> float:
    if greycomatrix_block.shape != _EXPECTED_BLOCK_SHAPE:
        raise ValueError(
            "Expected shape {0}, got {1}.".format(
                _EXPECTED_BLOCK_SHAPE, greycomatrix_block.shape
            )
        )
    return (
        np.sum(greycomatrix_block[:3, :3])
        + np.sum(greycomatrix_block[:2, :2])
        + 2.0 * greycomatrix_block[0, 0]
    )


_BlockSelector = Callable[[np.ndarray], np.ndarray]


def _darkness(greycomatrix: np.ndarray) -> np.ndarray:
    size = int(_DISCRETIZATION_LEVELS / 2)
    return greycomatrix[:size, :size]


def _lightness(greycomatrix: np.ndarray) -> np.ndarray:
    size = int(_DISCRETIZATION_LEVELS / 2 + 1)
    return greycomatrix[:-size:-1, :-size:-1]


def _structness_of(selector: _BlockSelector, directions: np.ndarray) -> float:
    return float(
        np.sum(_block_structness(selector(direction)) for direction in directions)
    )


def structness(image: np.ndarray, ignored: List = None) -> Tuple[float, float]:
    if ignored is None:
        ignored = []
    if len(image.shape) != 2:
        raise ValueError("Expected 2D image, got {0}D.".format(len(image.shape)))
    mask = _ignorance_mask(image, ignored)
    discrete = _discretize(image, mask)
    greycomatrix = _greycomatrix(discrete, mask)
    directions = _split_directions(greycomatrix)
    structness_of_darkness = _structness_of(_darkness, directions)
    structness_of_lightness = _structness_of(_lightness, directions)
    return structness_of_darkness, structness_of_lightness
