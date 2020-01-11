from functools import partial
from typing import Union

import numpy as np
from sklearn.base import clone

from divik._utils import Data, maybe_pool
from divik._seeding import seeded
from divik.sampler import UniformSampler, StratifiedSampler
from divik.score._gap import _sampled_dispersion as _dispersion


KMeans = 'divik.KMeans'


@seeded(wrapped_requires_seed=True)
def sampled_gap(data: Data, kmeans: KMeans,
                sample_size: Union[int, float] = 1000,
                n_jobs: int = None,
                seed: int = 0,
                n_trials: int = 100,
                return_deviation: bool = False) -> float:
    # TODO: Docs
    data_ = StratifiedSampler(n_rows=sample_size, n_samples=n_trials
                              ).fit(data, kmeans.labels_)
    reference_ = UniformSampler(n_rows=sample_size, n_samples=n_trials
                                ).fit(data)
    kmeans_ = clone(kmeans)
    with data_.parallel() as d, reference_.parallel() as r, \
            maybe_pool(n_jobs) as pool:
        compute_disp = partial(_dispersion, sampler=r, kmeans=kmeans_)
        ref_disp = pool.map(compute_disp, range(seed, seed + n_trials))
        compute_disp = partial(_dispersion, sampler=d, kmeans=kmeans_)
        data_disp = pool.map(compute_disp, range(seed, seed + n_trials))
    ref_disp = np.log(ref_disp)
    data_disp = np.log(data_disp)
    gap = np.mean(ref_disp) - np.mean(data_disp)
    result = (gap,)
    if return_deviation:
        std = np.sqrt(np.var(ref_disp) + np.var(data_disp)) / n_trials
        result += (std,)
    return result
