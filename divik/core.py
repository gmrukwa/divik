from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import ctypes
from functools import wraps
# RawArray exists, but PyCharm goes crazy
# noinspection PyUnresolvedReferences
from multiprocessing import Pool, RawArray
import os
import sys
from typing import Callable, Tuple, NamedTuple, List, Optional
import uuid

try:
    import gin
    from absl import flags
    _HAS_GIN = True
except ImportError:
    _HAS_GIN = False

import numpy as np
from skimage.color import label2rgb


__all__ = [
    'DivikResult',
    'normalize_rows',
    'maybe_pool',
    'share',
    'context_if',
    'seed',
    'seeded',
    'configurable',
    'parse_gin_args',
]

##############################################################################
# Types
##############################################################################

Table = np.ndarray  # 2D matrix
Data = Table
Centroids = Table
IntLabels = np.ndarray
SegmentationMethod = Callable[[Data], Tuple[IntLabels, Centroids]]
DivikResult = NamedTuple('DivikResult', [
    ('clustering', 'divik.AutoKMeans'),
    ('feature_selector', 'divik.feature_selection.StatSelectorMixin'),
    ('merged', IntLabels),
    ('subregions', List[Optional['DivikResult']]),
])


##############################################################################
# Utils
##############################################################################


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


##############################################################################
# Parallel
##############################################################################


class SharedArrayWrapper(metaclass=ABCMeta):
    @property
    @abstractmethod
    def value(self):
        """Get data (can be called by the recipient)

        Recipient must not modify the data.
        """
        pass


class SharedArray(metaclass=ABCMeta):
    @abstractmethod
    def store(self, data: np.ndarray) -> SharedArrayWrapper:
        """Store the data (must be called by the owner)"""
        pass

    @abstractmethod
    def purge(self):
        """Purge data (must be called by the owner)"""
        pass


class PosixSharedArrayWrapper(SharedArrayWrapper):
    def __init__(self, ref: str):
        self._ref = ref

    @property
    def value(self):
        return PosixSharedArray.DATA[self._ref]


class PosixSharedArray(SharedArray):
    DATA = {}

    def __init__(self):
        self._ref = str(uuid.uuid4())

    def store(self, data):
        if self._ref in PosixSharedArray.DATA:
            raise RuntimeError("UnixSharedArray already stores value")
        PosixSharedArray.DATA[self._ref] = data
        return PosixSharedArrayWrapper(self._ref)

    def purge(self):
        if self._ref in PosixSharedArray.DATA:
            del PosixSharedArray.DATA[self._ref]


class WinSharedArrayWrapper(SharedArrayWrapper):
    def __init__(self, shape, dtype, array):
        self._shape = shape
        self._dtype = dtype
        self._array = array

    @property
    def value(self):
        return np.frombuffer(self._array, dtype=self._dtype
                             ).reshape(self._shape)


class WinSharedArray(SharedArray):
    _DTYPES = {
        np.dtype(np.float64): ctypes.c_double,
        np.dtype(np.float32): ctypes.c_float,
        np.dtype(np.uint64): ctypes.c_uint64,
        np.dtype(np.uint32): ctypes.c_uint32,
        np.dtype(np.uint16): ctypes.c_uint16,
        np.dtype(np.int64): ctypes.c_int64,
        np.dtype(np.int32): ctypes.c_int32,
        np.dtype(np.int16): ctypes.c_int16,
    }

    def __init__(self):
        self._shape = None
        self._dtype = None
        self._array = None

    def store(self, data):
        if self._array is not None:
            raise RuntimeError("WindowsSharedArray already stores value")
        if data.dtype not in WinSharedArray._DTYPES:
            raise ValueError("Unsupported data type")
        self._shape = data.shape
        self._dtype = data.dtype
        ctype = WinSharedArray._DTYPES[data.dtype]
        self._array = RawArray(ctype, data.size)
        numpied = np.frombuffer(self._array, dtype=self._dtype)
        np.copyto(numpied, data.ravel())
        return WinSharedArrayWrapper(self._shape, self._dtype, self._array)

    def purge(self):
        self._shape = None
        self._dtype = None
        self._array = None


def _make_shared_array() -> SharedArray:
    if os.name == 'posix':
        return PosixSharedArray()
    return WinSharedArray()


@contextmanager
def share(array: np.ndarray):
    shared = _make_shared_array()
    try:
        yield shared.store(array)
    except Exception:
        raise
    finally:
        shared.purge()


def get_n_jobs(n_jobs):
    n_cpu = os.cpu_count() or 1
    n_jobs = 1 if n_jobs is None else n_jobs
    if n_jobs <= 0:
        n_jobs = min(n_jobs + 1 + n_cpu, n_cpu)
    n_jobs = n_jobs or n_cpu
    return n_jobs


class DummyPool:
    def __init__(self, processes, initializer=None, initargs=None,
                 *args, **kwargs):
        if initargs is None:
            initargs = ()
        if initializer is not None:
            initializer(*initargs)

    def apply(self, func, args, kwds):
        return func(*args, **kwds)

    # noinspection PyUnusedLocal
    def map(self, func, iterable, chunksize=None):
        return [func(v) for v in iterable]

    # noinspection PyUnusedLocal
    def starmap(self, func, iterable, chunksize=None):
        return [func(*v) for v in iterable]


@contextmanager
def maybe_pool(processes: int=None, *args, **kwargs):
    n_jobs = get_n_jobs(processes)
    if n_jobs == 1 or n_jobs == 0:
        yield DummyPool(n_jobs, *args, **kwargs)
    else:
        with Pool(n_jobs, *args, **kwargs) as pool:
            yield pool


##############################################################################
# Randomization maintenance
##############################################################################


@contextmanager
def seed(seed_: int = 0):
    """Crete seeded scope."""
    state = np.random.get_state()
    np.random.seed(seed_)
    yield
    np.random.set_state(state)


def seeded(wrapped_requires_seed: bool = False):
    """Create seeded scope for function call.

    Parameters
    ----------
    wrapped_requires_seed: bool, optional, default: False
        if true, passes seed parameter to the inner function
    """
    get = dict.get if wrapped_requires_seed else dict.pop

    def _seeded_maker(func):
        @wraps(func)
        def _seeded(*args, **kwargs):
            _seed = get(kwargs, 'seed', 0)
            with seed(_seed):
                return func(*args, **kwargs)
        return _seeded

    return _seeded_maker


##############################################################################
# gin-config compatibility
##############################################################################


def parse_gin_args():
    """Parse arguments with gin-config

    If you have `gin` extras installed, you can call `parse_gin_args`
    to parse command line arguments or config files to configure
    your runs.

    Command line arguments are used like `--param='DiviK.k_max=50'`.
    Config files are passed via `--config=path.gin`.

    More about format of `.gin` files can be found here:
    https://github.com/google/gin-config
    """
    import gin
    from absl import flags
    flags.DEFINE_multi_string(
        'config', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string(
        'param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    gin.parse_config_files_and_bindings(FLAGS.config, FLAGS.param)


def configurable(klass):
    """Marks class as configurable via gin-config"""
    if _HAS_GIN:
        klass.__init__ = gin.external_configurable(
            klass.__init__, name=klass.__name__, blacklist=['self'])
    return klass

