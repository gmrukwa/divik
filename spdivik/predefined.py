from functools import partial
from multiprocessing import Pool
from typing import Callable, Optional

import spdivik.distance as dst
import spdivik.divik as dv
import spdivik.feature_selection as fs
import spdivik.kmeans as km
import spdivik.score as sc
import spdivik.stop as st
import spdivik.types as ty

Divik = Callable[[ty.Data], Optional[dv.DivikResult]]


def _extreme_kmeans(metric: dst.KnownMetric,
                    iters_limit: int = 100) -> km.KMeans:
    distance = dst.ScipyDistance(metric)
    kmeans = km.KMeans(labeling=km.Labeling(distance),
                       initialize=km.ExtremeInitialization(distance),
                       number_of_iterations=iters_limit)
    return kmeans


def _dunn_optimized_kmeans(metric: dst.KnownMetric,
                           iters_limit: int = 100) -> sc.Optimizer:
    distance = dst.ScipyDistance(metric)
    dunn = partial(sc.dunn, distance=distance)
    kmeans = _extreme_kmeans(metric, iters_limit)
    sweep_clusters_number = [
        sc.ParameterValues('number_of_clusters', list(range(2, 11)))
    ]
    best_kmeans_with_dunn = sc.Optimizer(score=dunn,
                                         segmentation_method=kmeans,
                                         parameters=sweep_clusters_number)
    return best_kmeans_with_dunn


_AMPLITUDE_FILTER = dv.FilteringMethod(
    'amplitude',
    partial(fs.select_by,
            statistic=fs.amplitude,
            discard_up_to=1,
            # discarding only lowest component, if possible
            preserve_topmost=True,
            min_features_percentage=.05))
_VARIANCE_FILTER = dv.FilteringMethod(
    'variance',
    partial(fs.select_by,
            statistic=fs.variance,
            # selecting only most varying component, if possible
            discard_up_to=-1,
            preserve_topmost=True,
            min_features_percentage=.05))


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


def proteomic(minimal_split_segment: int = 20, iters_limit: int = 100) -> Divik:
    best_kmeans_with_dunn = _dunn_optimized_kmeans(
        dst.KnownMetric.correlation, iters_limit)
    stop_for_small_size = partial(st.minimal_size, size=minimal_split_segment)
    divik = partial(dv.divik,
                    split=best_kmeans_with_dunn,
                    feature_selectors=[_VARIANCE_FILTER],
                    stop_condition=stop_for_small_size)
    prefiltered_divik = _PrefilteringWrapper(prefilter=_AMPLITUDE_FILTER,
                                             divik=divik)
    return prefiltered_divik


def master(gap_trials: int=100, iters_limit: int = 100, pool: Pool=None) -> Divik:
    metric = dst.KnownMetric.correlation
    best_kmeans_with_dunn = _dunn_optimized_kmeans(metric, iters_limit)
    fast_kmeans = partial(_extreme_kmeans(metric, iters_limit=10),
                          number_of_clusters=2)
    distance = dst.ScipyDistance(metric)
    stop_if_split_makes_no_sense = st.combine(
        partial(st.minimal_size, size=20),
        st.Gap(distance, fast_kmeans, gap_trials, pool=pool))
    divik = partial(dv.divik,
                    split=best_kmeans_with_dunn,
                    feature_selectors=[_AMPLITUDE_FILTER, _VARIANCE_FILTER],
                    stop_condition=stop_if_split_makes_no_sense)
    return divik
