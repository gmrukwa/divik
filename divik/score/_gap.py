import logging
from functools import partial

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from sklearn.base import clone

from divik.core import (
    Data,
    maybe_pool,
    normalize_rows,
    seeded,
)
from divik.sampler import BaseSampler, UniformSampler

KMeans = "divik.KMeans"
_BIG_PRIME = 54673


def _dispersion(data: Data, kmeans: KMeans) -> float:
    assert data.shape[0] == kmeans.labels_.size, "kmeans not fit on this data"
    if kmeans.normalize_rows:
        data = normalize_rows(data)
    clusters = pd.DataFrame(data).groupby(kmeans.labels_)
    return float(
        np.mean(
            [
                np.mean(dist.pdist(cluster_members.values, kmeans.distance))
                for _, cluster_members in clusters
                if cluster_members.shape[0] != 1
            ]
        )
    )


def _sampled_dispersion(
    seed: int, sampler: BaseSampler, kmeans: KMeans, fit: bool = True
) -> float:
    logging.debug(f"Sampling with seed {seed}.")
    X = sampler.get_sample(seed)
    logging.debug(f"Sample shape {X.shape}")
    if kmeans.normalize_rows:
        logging.debug("Normalizing rows.")
        X = normalize_rows(X)
    if fit:
        logging.debug("Fitting kmeans for sample.")
        y = kmeans.fit_predict(X)
    else:
        logging.debug("Predicting labels for sample.")
        y = kmeans.predict(X)
    logging.debug("Computing dispersion for clustered sample.")
    clusters = pd.DataFrame(X).groupby(y)
    return float(
        np.mean(
            [
                np.mean(dist.pdist(cluster_members.values, kmeans.distance))
                for _, cluster_members in clusters
                if cluster_members.shape[0] != 1
            ]
        )
    )


def gap(
    data: Data,
    kmeans: KMeans,
    n_jobs: int = None,
    seed: int = 0,
    n_trials: int = 100,
    return_deviation: bool = False,
) -> float:
    reference_ = UniformSampler(n_rows=None, n_samples=n_trials).fit(data)
    kmeans_ = clone(kmeans)
    seeds = list(seed + np.arange(n_trials) * _BIG_PRIME)
    with reference_.parallel() as r:
        with maybe_pool(n_jobs, initializer=r.initializer, initargs=r.initargs) as pool:
            compute_disp = partial(_sampled_dispersion, sampler=r, kmeans=kmeans_)
            ref_disp = pool.map(compute_disp, seeds)
    ref_disp = np.log(ref_disp)
    data_disp = np.log(_dispersion(data, kmeans))
    gap = np.mean(ref_disp) - data_disp
    result = (gap,)
    if return_deviation:
        std = np.sqrt(1 + 1 / n_trials) * np.std(ref_disp)
        result += (std,)
    return result
