from contextlib import contextmanager
import inspect
# RawArray exists, but PyCharm goes crazy
# noinspection PyUnresolvedReferences
from multiprocessing import Pool, RawArray

import numpy as np
from skimage.color import label2rgb

from ._types import Data


def normalize_rows(data: Data) -> Data:
    normalized = data - data.mean(axis=1)[:, np.newaxis]
    norms = np.sum(np.abs(normalized) ** 2, axis=-1, keepdims=True)**(1./2)
    normalized /= norms
    return normalized


def visualize(label, xy, shape=None):
    x, y = xy.T
    if shape is None:
        shape = np.max(y) + 1, np.max(x) + 1
    y = y.max() - y
    label = label - label.min() + 1
    label_map = np.zeros(shape, dtype=int)
    label_map[y, x] = label
    image = label2rgb(label_map, bg_label=0)
    return image


@contextmanager
def context_if(condition, context, *args, **kwargs):
    if condition:
        with context(*args, **kwargs) as c:
            yield c
    else:
        yield None


def build(klass, **kwargs):
    """Build instance of klass using matching kwargs"""
    known_param_names = list(inspect.signature(klass).parameters.keys())
    known_params = {n: kwargs[n] for n in known_param_names if n in kwargs}
    return klass(**known_params)
