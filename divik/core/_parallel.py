"""multiprocessing.Pool helpers"""
import ctypes
import os
import uuid
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

# RawArray exists, but PyCharm goes crazy
# noinspection PyUnresolvedReferences
from multiprocessing import Pool, RawArray

import numpy as np


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
        return np.frombuffer(self._array, dtype=self._dtype).reshape(self._shape)


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
    if os.name == "posix":
        return PosixSharedArray()
    return WinSharedArray()


@contextmanager
def share(array: np.ndarray):
    """Share a numpy array between ``multiprocessing.Pool`` processes"""
    shared = _make_shared_array()
    try:
        yield shared.store(array)
    except Exception:
        raise
    finally:
        shared.purge()


def get_n_jobs(n_jobs):
    """Determine the actual number of possible jobs"""
    n_cpu = os.cpu_count() or 1
    n_jobs = 1 if n_jobs is None else n_jobs
    if n_jobs <= 0:
        n_jobs = min(n_jobs + 1 + n_cpu, n_cpu)
    n_jobs = n_jobs or n_cpu
    return n_jobs


class DummyPool:
    def __init__(self, processes, initializer=None, initargs=None, *args, **kwargs):
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
def maybe_pool(processes: int = None, *args, **kwargs):
    """Create ``multiprocessing.Pool`` if multiple CPUs are allowed

    Examples
    --------

    >>> from divik.core import maybe_pool
    >>> with maybe_pool(processes=1) as pool:
    ...     # Runs in sequential
    ...     pool.map(id, range(10000))
    >>> with maybe_pool(processes=-1) as pool:
    ...     # Runs with all cores
    ...     pool.map(id, range(10000))
    """
    n_jobs = get_n_jobs(processes)
    if n_jobs == 1 or n_jobs == 0:
        yield DummyPool(n_jobs, *args, **kwargs)
    else:
        with Pool(n_jobs, *args, **kwargs) as pool:
            yield pool
