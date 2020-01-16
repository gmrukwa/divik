import gc
from functools import partial
from typing import Optional

import numpy as np
from sklearn import clone

from divik.core import Data, DivikResult
from ._report import DivikReporter


def _recursive_selection(current_selection: np.ndarray, partition: np.ndarray,
                         cluster_number: int) -> np.ndarray:
    selection = np.zeros(shape=current_selection.shape, dtype=bool)
    selection[current_selection] = partition == cluster_number
    return selection


StatSelector = 'divik.feature_selection.StatSelectorMixin'
DunnSearch = 'divik.cluster._kmeans._dunn.DunnSearch'
GAPSearch = 'divik.cluster._kmeans._gap.GAPSearch'


# @gmrukwa: I could not find more readable solution than recursion for now.
def dunn_divik(data: Data, selection: np.ndarray,
               fast_kmeans: GAPSearch, full_kmeans: DunnSearch,
               feature_selector: StatSelector,
               minimal_size: int, rejection_size: int, report: DivikReporter) \
        -> Optional[DivikResult]:
    subset = data[selection]

    if subset.shape[0] <= max(full_kmeans.max_clusters, minimal_size):
        report.finished_for(subset.shape[0])
        return None

    report.filter(subset)
    feature_selector = clone(feature_selector)
    filtered_data = feature_selector.fit_transform(subset)
    report.filtered(filtered_data)

    report.stop_check()
    fast_kmeans = clone(fast_kmeans).fit(filtered_data)
    if fast_kmeans.fitted_ and fast_kmeans.n_clusters_ == 1:
        report.finished_for(subset.shape[0])
        return None

    report.processing(filtered_data)
    clusterer = clone(full_kmeans).fit(filtered_data)
    partition = clusterer.labels_
    _, counts = np.unique(partition, return_counts=True)

    if any(counts <= rejection_size):
        report.rejected(subset.shape[0])
        return None

    report.recurring(len(counts))
    recurse = partial(
        dunn_divik, data=data, fast_kmeans=fast_kmeans,
        full_kmeans=full_kmeans, feature_selector=feature_selector,
        minimal_size=minimal_size, rejection_size=rejection_size,
        report=report)
    del subset
    del filtered_data
    gc.collect()
    subregions = [
        recurse(selection=_recursive_selection(selection, partition, cluster))
        for cluster in np.unique(partition)
    ]

    report.assemble()
    return DivikResult(clustering=clusterer, feature_selector=feature_selector,
                       merged=partition, subregions=subregions)