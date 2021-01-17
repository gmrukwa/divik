import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager

# RawArray exists, but PyCharm goes crazy
# noinspection PyUnresolvedReferences
from multiprocessing import Pool, RawArray

import numpy as np
import tqdm
from skimage.color import label2rgb

from .. import __version__
from ._types import Data


def normalize_rows(data: Data) -> Data:
    """Translate and scale rows to zero mean and vector length equal one"""
    normalized = data - data.mean(axis=1)[:, np.newaxis]
    norms = np.sum(np.abs(normalized) ** 2, axis=-1, keepdims=True) ** (1.0 / 2)
    normalized /= norms
    return normalized


def visualize(label, xy, shape=None):
    """Create RGB map of labels over with given coordinates"""
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
    """Create context with given params only if the condition is ``True``"""
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


def prepare_destination(
    destination: str, omit_datetime: bool = False, exist_ok: bool = False
) -> str:
    if not omit_datetime:
        datetime = time.strftime("%Y%m%d-%H%M%S")
        destination = os.path.join(destination, datetime)
    os.makedirs(destination, exist_ok=exist_ok)
    return destination


_PRECISE_FORMAT = (
    "%(asctime)s [%(levelname)s] %(filename)40s:%(lineno)3s"
    + " - %(funcName)40s\t%(message)s"
)
_INFO_FORMAT = "%(asctime)s [%(levelname)s]\t%(message)s"


def _file_handler(destination: str):
    log_destination = os.path.join(destination, "logs.txt")
    handler = logging.FileHandler(filename=log_destination, mode="a")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_PRECISE_FORMAT))
    return handler


def _stream_handler(verbose: bool):
    handler = logging.StreamHandler(tqdm.tqdm)
    if verbose:
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(_INFO_FORMAT))
    else:
        handler.setLevel(logging.CRITICAL)
        handler.setFormatter(logging.Formatter(_INFO_FORMAT))
    return handler


def setup_logger(destination: str, verbose: bool = False):
    stream_handler = _stream_handler(verbose)
    file_handler = _file_handler(destination)
    handlers = [
        stream_handler,
        file_handler,
    ]
    del logging.root.handlers[:]
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)
    version_notice = "Using " + sys.argv[0] + " (divik, version " + __version__ + ")"
    logging.info(version_notice)
