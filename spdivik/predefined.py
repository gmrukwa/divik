from functools import partial
from multiprocessing import Pool
from typing import Callable, Optional

from tqdm import tqdm

import spdivik.distance as dst
import spdivik.divik as dv
import spdivik.feature_selection as fs
import spdivik.kmeans as km
import spdivik.score as sc
import spdivik.stop as st
import spdivik.types as ty

Divik = Callable[[ty.Data], Optional[dv.DivikResult]]


def _dunn_optimized_kmeans(distance: dst.DistanceMetric,
                           kmeans: km.KMeans,
                           pool: Pool=None,
                           k_max: int=10) -> sc.Optimizer:
    dunn = partial(sc.dunn, distance=distance)
    sweep_clusters_number = [
        sc.ParameterValues('number_of_clusters', list(range(2, k_max + 1)))
    ]
    best_kmeans_with_dunn = sc.Optimizer(score=dunn,
                                         segmentation_method=kmeans,
                                         parameters=sweep_clusters_number,
                                         pool=pool)
    return best_kmeans_with_dunn


_AMPLITUDE_FILTER = dv.FilteringMethod(
    'amplitude',
    partial(fs.select_by,
            statistic=fs.amplitude,
            discard_up_to=1,
            # discarding only lowest component, if possible
            preserve_topmost=True))
_VARIANCE_FILTER = dv.FilteringMethod(
    'variance',
    partial(fs.select_by,
            statistic=fs.variance,
            # selecting only most varying component, if possible
            discard_up_to=-1,
            preserve_topmost=True))


class _PrefilteringWrapper:
    def __init__(self, prefilter: dv.FilteringMethod, divik: Divik):
        self._prefilter = prefilter
        self._divik = divik

    def __call__(self, data: ty.Data) -> dv.DivikResult:
        preselection, threshold = self._prefilter(data)
        result = self._divik(data[:, preselection])
        result.thresholds[self._prefilter.name] = threshold
        result.filters[self._prefilter.name] = preselection
        return result


def proteomic(minimal_split_segment: int = 20, iters_limit: int = 100,
              progress_reporter: tqdm=None, pool: Pool=None) -> Divik:
    distance = dst.ScipyDistance(dst.KnownMetric.correlation)
    kmeans = km.KMeans(labeling=km.Labeling(distance),
                       initialize=km.ExtremeInitialization(distance),
                       number_of_iterations=iters_limit)
    best_kmeans_with_dunn = _dunn_optimized_kmeans(
        distance, kmeans, pool=pool)
    stop_for_small_size = partial(st.minimal_size, size=minimal_split_segment)
    divik = partial(dv.divik,
                    split=best_kmeans_with_dunn,
                    feature_selectors=[_VARIANCE_FILTER],
                    stop_condition=stop_for_small_size,
                    progress_reporter=progress_reporter,
                    min_features_percentage=.05)
    prefiltered_divik = _PrefilteringWrapper(prefilter=_AMPLITUDE_FILTER,
                                             divik=divik)
    return prefiltered_divik


def master(gap_trials: int=100, distance_percentile: float=99.,
           iters_limit: int = 100, pool: Pool=None,
           progress_reporter: tqdm=None,
           distance: dst.DistanceMetric=None) -> Divik:
    assert 0 <= distance_percentile <= 100, distance_percentile
    if distance is None:
        distance = dst.SpearmanDistance()
    labeling = km.Labeling(distance)
    initialize = km.PercentileInitialization(distance, distance_percentile)
    kmeans = km.KMeans(labeling=km.Labeling(distance),
                       initialize=initialize,
                       number_of_iterations=iters_limit)
    best_kmeans_with_dunn = _dunn_optimized_kmeans(distance, kmeans, pool)
    fast_kmeans = partial(km.KMeans(labeling=labeling,
                                    initialize=initialize,
                                    number_of_iterations=10),
                          number_of_clusters=2)
    stop_if_split_makes_no_sense = st.combine(
        partial(st.minimal_size, size=20),
        st.Gap(distance, fast_kmeans, gap_trials, pool=pool))
    divik = partial(dv.divik,
                    split=best_kmeans_with_dunn,
                    feature_selectors=[_AMPLITUDE_FILTER, _VARIANCE_FILTER],
                    stop_condition=stop_if_split_makes_no_sense,
                    progress_reporter=progress_reporter,
                    min_features_percentage=.05)
    return divik
