from functools import partial
import gc
from multiprocessing import Pool
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

from divik._score._picker import Picker
from divik._utils import Data, DummyPool, normalize_rows, maybe_pool
from divik._seeding import seeded


KMeans = 'divik.KMeans'


def _dispersion(data: Data, kmeans: KMeans) -> float:
    assert data.shape[0] == kmeans.labels_.size, "kmeans not fit on this data"
    if kmeans.normalize_rows:
        data = normalize_rows(data)
    clusters = pd.DataFrame(data).groupby(kmeans.labels_)
    return float(np.sum([
        np.sum(dist.cdist(kmeans.cluster_centers_[np.newaxis, label],
                          cluster_members.values, kmeans.distance))
        for label, cluster_members in clusters
    ]))


def _dispersion_of_random_sample(seed: int,
                                 shape: Tuple[int, int],
                                 scale: MinMaxScaler,
                                 kmeans: KMeans) -> float:
    np.random.seed(seed)
    sample = scale.inverse_transform(np.random.random_sample(shape))
    dispersion = _dispersion(sample, kmeans.fit(sample))
    del sample
    gc.collect()
    return dispersion


@seeded(wrapped_requires_seed=True)
def gap(data: Data, kmeans: KMeans, seed: int = 0, n_trials: int = 100,
        pool: Pool = DummyPool(), return_deviation: bool = False,
        max_iter: int = 10) -> float:
    scale = MinMaxScaler().fit(data)
    fast_kmeans = clone(kmeans)
    fast_kmeans.max_iter = max_iter
    compute_dispersion = partial(_dispersion_of_random_sample,
                                 shape=data.shape,
                                 scale=scale,
                                 kmeans=fast_kmeans)
    dispersions = pool.map(compute_dispersion, range(seed, seed + n_trials))
    reference = _dispersion(data, kmeans)
    log_dispersions = np.log(dispersions)
    gap_value = np.mean(log_dispersions) - np.log(reference)
    result = (gap_value, )
    if return_deviation:
        standard_deviation = np.sqrt(1 + 1 / n_trials) \
                             * np.std(log_dispersions)
        result += (standard_deviation,)
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
        with maybe_pool(self.n_jobs) as pool:
            gap_ = partial(gap, data, seed=self.seed, n_trials=self.n_trials,
                           return_deviation=True, pool=pool,
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
