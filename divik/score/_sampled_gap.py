import logging
from functools import partial
from typing import Union

import numpy as np
from sklearn.base import clone

from divik.core import Data, maybe_pool
from divik.sampler import StratifiedSampler, UniformSampler
from divik.score._gap import _sampled_dispersion as _dispersion

KMeans = "divik.cluster.KMeans"
_BIG_PRIME = 40013


def _pool_initialize(initializers, *args):
    for arg, sampler in zip(args, initializers):
        sampler.initializer(*arg)


def sampled_gap(
    data: Data,
    kmeans: KMeans,
    sample_size: Union[int, float] = 1000,
    n_jobs: int = None,
    seed: int = 0,
    n_trials: int = 100,
    return_deviation: bool = False,
) -> float:
    # TODO: Docs
    logging.debug("Creating samplers.")
    data_ = StratifiedSampler(n_rows=sample_size, n_samples=n_trials).fit(
        data, kmeans.labels_
    )
    reference_ = UniformSampler(n_rows=sample_size, n_samples=n_trials).fit(data)
    kmeans_ = clone(kmeans)
    seeds = list(seed + np.arange(n_trials) * _BIG_PRIME)
    logging.debug(f"Generated seeds: {seeds}.")
    logging.debug(f"Entering parallel context with n_jobs={n_jobs}.")
    with data_.parallel() as d, reference_.parallel() as r:
        initializer = partial(_pool_initialize, [d, r])
        with maybe_pool(
            n_jobs, initializer=initializer, initargs=(d.initargs, r.initargs)
        ) as pool:
            logging.debug("Computing reference dispersion.")
            compute_disp = partial(_dispersion, sampler=r, kmeans=kmeans_)
            ref_disp = pool.map(compute_disp, seeds)
            logging.debug("Computing data dispersion.")
            compute_disp = partial(_dispersion, sampler=d, kmeans=kmeans, fit=False)
            data_disp = pool.map(compute_disp, seeds)
    logging.debug("Left parallel context.")
    ref_disp = np.log(ref_disp)
    data_disp = np.log(data_disp)
    gap = np.mean(ref_disp) - np.mean(data_disp)
    result = (gap,)
    if return_deviation:
        std = np.sqrt(np.var(ref_disp) + np.var(data_disp)) / n_trials
        result += (std,)
    return result
