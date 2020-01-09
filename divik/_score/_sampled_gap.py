from functools import partial
from typing import List, Union

import numpy as np
from sklearn.base import clone

from divik._utils import Data, maybe_pool
from divik._seeding import seeded
from divik.sampler import UniformSampler, StratifiedSampler
from divik._score._gap import _sampled_dispersion as _dispersion, GapPicker


KMeans = 'divik.KMeans'


@seeded(wrapped_requires_seed=True)
def sampled_gap(data: Data, kmeans: KMeans,
                sample_size: Union[int, float] = 1000,
                n_jobs: int = None,
                seed: int = 0,
                n_trials: int = 100,
                return_deviation: bool = False,
                max_iter: int = 10) -> float:
    # TODO: Docs
    # TODO: Tests
    data_ = StratifiedSampler(n_rows=sample_size, n_samples=n_trials
                              ).fit(data, kmeans.labels_)
    reference_ = UniformSampler(n_rows=sample_size, n_samples=n_trials
                                ).fit(data)
    kmeans_ = clone(kmeans)
    kmeans_.max_iter = max_iter
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


class SamplingGapPicker(GapPicker):
    def __init__(self, sample_size: int = 1000, max_iter: int = 10,
                 seed: int = 0, n_trials: int = 10,
                 correction: bool = True, n_jobs: int = 1):
        super().__init__(
            max_iter=max_iter, seed=seed, n_trials=n_trials,
            correction=correction, n_jobs=n_jobs)
        self.sample_size = sample_size

    def score(self, data: Data, estimators: List[KMeans]) -> np.ndarray:
        gap_ = partial(sampled_gap, data,
                       sample_size=self.sample_size,
                       n_jobs=self.n_jobs,
                       seed=self.seed,
                       n_trials=self.n_trials,
                       return_deviation=True,
                       max_iter=self.max_iter)
        scores = [gap_(kmeans=estimator) for estimator in estimators]
        return np.array(scores)
