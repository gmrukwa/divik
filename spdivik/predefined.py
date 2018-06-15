from functools import partial
from typing import Callable, Optional

from functional import pipe

import spdivik.distance as dst
import spdivik.divik as dv
import spdivik.feature_selection as fs
import spdivik.kmeans as km
import spdivik.score as sc
import spdivik.stop as st
import spdivik.types as ty

Divik = Callable[[ty.Data], Optional[dv.DivikResult]]


def _dunn_optimized_kmeans(metric: dst.KnownMetric,
                           iters_limit: int=100) -> sc.Optimizer:
    distance = dst.ScipyDistance(metric)
    dunn = partial(sc.dunn, distance=distance)
    kmeans = km.KMeans(labeling=km.Labeling(distance),
                       initialize=km.ExtremeInitialization(distance),
                       number_of_iterations=iters_limit)
    sweep_clusters_number = [
        sc.ParameterValues('number_of_clusters', list(range(2, 11)))
    ]
    best_kmeans_with_dunn = sc.Optimizer(score=dunn,
                                         segmentation_method=kmeans,
                                         parameters=sweep_clusters_number)
    return best_kmeans_with_dunn


_PROTEOMIC_DENOISE_SETTING = {
    'statistic': fs.amplitude,
    'discard_up_to': 1,  # discarding only lowest component, if possible
    'preserve_topmost': True,
    'min_features_percentage': .05
}
_PROTEOMIC_FILTER_SETTING = {
    'statistic': fs.variance,
    'discard_up_to': -1,  # selecting only most varying component, if possible
    'preserve_topmost': True,
    'min_features_percentage': .05
}


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


def proteomic(minimal_split_segment: int = 20, iters_limit: int=100) -> Divik:
    amplitude_filter = partial(fs.select_by, **_PROTEOMIC_DENOISE_SETTING)
    noise_removal_strategy = dv.FilteringMethod('amplitude', amplitude_filter)

    best_kmeans_with_dunn = _dunn_optimized_kmeans(
        dst.KnownMetric.correlation, iters_limit)
    variance_filter = partial(fs.select_by, **_PROTEOMIC_FILTER_SETTING)
    variance_strategy = dv.FilteringMethod('variance', variance_filter)
    stop_for_small_size = partial(st.minimal_size, size=minimal_split_segment)
    divik = partial(dv.divik,
                    split=best_kmeans_with_dunn,
                    feature_selectors=[variance_strategy],
                    stop_condition=stop_for_small_size)

    prefiltered_divik = _PrefilteringWrapper(prefilter=noise_removal_strategy,
                                             divik=divik)
    return prefiltered_divik
