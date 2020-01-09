from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from sklearn.base import clone

from divik._score._picker import Picker
from divik._utils import Data, normalize_rows, maybe_pool
from divik._seeding import seeded
from divik.sampler import BaseSampler, UniformSampler


KMeans = 'divik.KMeans'


def _dispersion(data: Data, kmeans: KMeans) -> float:
    assert data.shape[0] == kmeans.labels_.size, "kmeans not fit on this data"
    if kmeans.normalize_rows:
        data = normalize_rows(data)
    clusters = pd.DataFrame(data).groupby(kmeans.labels_)
    return float(np.mean([
        np.mean(dist.pdist(cluster_members.values, kmeans.distance))
        for _, cluster_members in clusters
    ]))


def _sampled_dispersion(seed: int, sampler: BaseSampler, kmeans: KMeans) \
        -> float:
    X = sampler.get_sample(seed)
    if kmeans.normalize_rows:
        X = normalize_rows(X)
    y = kmeans.fit_predict(X)
    clusters = pd.DataFrame(X).groupby(y)
    return float(np.mean([
        np.mean(dist.pdist(cluster_members.values, kmeans.distance))
        for _, cluster_members in clusters
    ]))


@seeded(wrapped_requires_seed=True)
def gap(data: Data, kmeans: KMeans,
        n_jobs: int = None,
        seed: int = 0,
        n_trials: int = 100,
        return_deviation: bool = False,
        max_iter: int = 10) -> float:
    reference_ = UniformSampler(n_rows=None, n_samples=n_trials
                                ).fit(data)
    kmeans_ = clone(kmeans)
    kmeans_.max_iter = max_iter
    with reference_.parallel() as r, maybe_pool(n_jobs) as pool:
        compute_disp = partial(_sampled_dispersion, sampler=r, kmeans=kmeans_)
        ref_disp = pool.map(compute_disp, range(seed, seed + n_trials))
    ref_disp = np.log(ref_disp)
    data_disp = np.log(_dispersion(data, kmeans))
    gap = np.mean(ref_disp) - data_disp
    result = (gap,)
    if return_deviation:
        std = np.sqrt(1 + 1 / n_trials) * np.std(ref_disp)
        result += (std,)
    return result


class GapPicker(Picker):
    def __init__(self, max_iter: int = 10, seed: int = 0, n_trials: int = 10,
                 correction: bool = True, n_jobs: int = 1):
        super().__init__(n_jobs=n_jobs)
        self.max_iter = max_iter
        self.seed = seed
        self.n_trials = n_trials
        self.correction = correction

    def score(self, data: Data, estimators: List[KMeans]) -> np.ndarray:
        gap_ = partial(gap, data,
                       n_jobs=self.n_jobs,
                       seed=self.seed,
                       n_trials=self.n_trials,
                       return_deviation=True,
                       max_iter=self.max_iter)
        scores = [gap_(kmeans=estimator) for estimator in estimators]
        return np.array(scores)

    def select(self, scores: np.ndarray) -> Optional[int]:
        GAP = scores[:, 0]
        s_k = scores[:, 1]
        if self.correction:
            is_suggested = GAP[:-1] > (GAP[1:] + s_k[1:])
            suggested_locations = list(np.flatnonzero(is_suggested))
        else:
            suggested_locations = [int(np.argmax(GAP))]
        return suggested_locations[0] if suggested_locations else None

    def report(self, estimators: List[KMeans], scores: np.ndarray) \
            -> pd.DataFrame:
        GAP = scores[:, 0]
        s_k = scores[:, 1]
        best = self.select(scores)
        suggested = np.zeros((len(estimators) - 1,), dtype=bool)
        if best is not None:
            suggested[best] = True
        suggested = list(suggested)
        suggested.append(None)
        return pd.DataFrame(
            data={
                'number_of_clusters': [e.n_clusters for e in estimators],
                'GAP': GAP,
                's_k': s_k,
                'suggested_number_of_clusters': suggested
            },
            columns=[
                'number_of_clusters',
                'GAP',
                's_k',
                'suggested_number_of_clusters'
            ]
        )
