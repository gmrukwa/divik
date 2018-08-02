import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import spdivik.rejection as rj
import spdivik.types as ty


def depth(tree, children_collection_name='subregions'):
    if tree is None:
        return 1
    return max(depth(subtree) for subtree in getattr(tree, children_collection_name)) + 1


def total_number_of_clusters(tree) -> int:
    if tree is None:
        return 1
    return sum(total_number_of_clusters(subtree) for subtree in tree.subregions)


def merged_partition(tree: ty.DivikResult, levels_limit: int=np.inf) \
        -> ty.IntLabels:
    return _merged_partition(tree.partition, tree.subregions, levels_limit)


def _merged_partition(partition: ty.IntLabels,
                      subregions: List[Optional[ty.DivikResult]],
                      levels_limit: int=np.inf) \
        -> ty.IntLabels:
    result = partition * 0 - 1
    known_clusters = 0
    for cluster_number, subregion in enumerate(subregions):
        current_cluster = partition == cluster_number
        if subregion is None or levels_limit <= 1:
            result[current_cluster] = known_clusters
            known_clusters += 1
        else:
            local_partition = _merged_partition(subregion.partition,
                                                subregion.subregions,
                                                levels_limit-1)
            result[current_cluster] = local_partition + known_clusters
            known_clusters += np.max(local_partition) + 1
    return result


def _update_graph(tree, size: int, graph: nx.Graph = None, parent=None):
    tree_node = len(graph)
    graph.add_node(tree_node, size=size)
    if parent is not None:
        graph.add_edge(parent, tree_node)
    if tree is None:
        return
    for idx, subtree in enumerate(tree.subregions):
        _update_graph(subtree, size=np.sum(tree.partition == idx), graph=graph,
                      parent=tree_node)


def _as_graph(tree) -> nx.DiGraph:
    graph = nx.DiGraph()
    _update_graph(tree, size=tree.partition.size, graph=graph, parent=None)
    return graph


def _make_labels(graph):
    attributes = nx.get_node_attributes(graph, 'size')
    return {
        idx: "{0} (size={1})".format(idx, attributes[idx])
        for idx in graph
    }


def _make_sizes(graph):
    values = np.array(list(nx.get_node_attributes(graph, 'size').values()))
    return list(np.sqrt(values))


def scale_plot_size(factor=1.5):
    import matplotlib as mpl
    default_dpi = mpl.rcParamsDefault['figure.dpi']
    mpl.rcParams['figure.dpi'] = default_dpi * factor


def _make_positions(graph):
    intermediate = nx.spectral_layout(graph)
    return nx.spring_layout(graph, pos=intermediate, iterations=20)


def plot(tree, with_size=False):
    graph = _as_graph(tree)
    arguments = {
        'G': graph,
        'pos': _make_positions(graph),
        'node_size': _make_sizes(graph),
        'font_size': 3,
        'width': .1  # lines width
    }
    if with_size:
        arguments['labels'] = _make_labels(graph)
    nx.draw_networkx(**arguments)
    plt.axis('off')


def dice(first, second):
    numerator = 2. * np.sum(np.logical_and(first, second))
    denominator = np.sum(first) + np.sum(second)
    return numerator / denominator


def positive_predictive_value(first, second):
    true_positive = np.logical_and(first, second).sum()
    false_positive = np.logical_and(first, ~second).sum()
    return float(true_positive) / (true_positive + false_positive)


def true_positive_rate(first, second):
    true_positive = np.logical_and(first, second).sum()
    false_negative = np.logical_and(~first, second).sum()
    return float(true_positive) / (true_positive + false_negative)


def statistic(merged, diagnoses, func):
    clusters = np.unique(merged)
    diagnosis_types = np.unique(diagnoses)
    return pd.DataFrame({
        diagnosis: [
            func(merged == cluster, diagnoses == diagnosis)
            for cluster in clusters
        ]
        for diagnosis in diagnosis_types
    })


def reject_split(tree: Optional[ty.DivikResult],
                 rejection_conditions: List[rj.RejectionCondition]) \
        -> Optional[ty.DivikResult]:
    if tree is None:
        logging.debug("Rejecting empty.")
        return None
    segmentation = (tree.partition, tree.centroids, tree.quality)
    if any(reject(segmentation) for reject in rejection_conditions):
        logging.debug("Rejecting by condition.")
        return None
    allowed_subregions = [
        reject_split(subregion, rejection_conditions)
        for subregion in tree.subregions
    ]
    merged = _merged_partition(tree.partition, allowed_subregions)
    logging.debug("Returning pruned tree.")
    return ty.DivikResult(
        centroids=tree.centroids,
        quality=tree.quality,
        partition=tree.partition,
        filters=tree.filters,
        thresholds=tree.thresholds,
        merged=merged,
        subregions=allowed_subregions
    )
