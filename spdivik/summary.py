import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import spdivik.divik as dv
import spdivik.types as ty


def depth(tree, children_collection_name='subregions'):
    if tree is None:
        return 1
    return max(depth(subtree) for subtree in getattr(tree, children_collection_name)) + 1


def total_number_of_clusters(tree) -> int:
    if tree is None:
        return 1
    return sum(total_number_of_clusters(subtree) for subtree in tree.subregions)


def merged_partition(tree: dv.DivikResult) -> ty.IntLabels:
    partition = tree.partition * 0 - 1
    known_clusters = 0
    for cluster_number, subregion in enumerate(tree.subregions):
        current_cluster = tree.partition == cluster_number
        if subregion is None:
            partition[current_cluster] = known_clusters
            known_clusters += 1
        else:
            local_partition = merged_partition(subregion)
            partition[current_cluster] = local_partition + known_clusters
            known_clusters += np.max(local_partition) + 1
    return partition


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
