"""Summarizing functions for DiviK result.

_summary.py

Copyright 2019 Grzegorz Mrukwa

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

import divik.core as u


def depth(tree, children_collection_name="subregions"):
    """Get tree depth."""
    if tree is None:
        return 1
    return (
        max(depth(subtree) for subtree in getattr(tree, children_collection_name)) + 1
    )


def total_number_of_clusters(tree) -> int:
    """Get the number of leaves in the tree."""
    if tree is None:
        return 1
    return sum(total_number_of_clusters(subtree) for subtree in tree.subregions)


def merged_partition(
    tree: u.DivikResult, levels_limit: int = np.inf, return_paths: bool = False
) -> u.IntLabels:
    """Compute merged segmentation labels."""
    assert tree is not None, "Nothing was segmented."
    merged, paths = _merged_partition(
        tree.clustering.labels_, tree.subregions, levels_limit
    )
    if return_paths:
        return merged, paths
    return merged


def _merged_partition(
    partition: u.IntLabels,
    subregions: List[Optional[u.DivikResult]],
    levels_limit: int = np.inf,
) -> Tuple[u.IntLabels, Dict[int, Tuple[int]]]:
    """Compute merged segmentation labels."""
    result = partition * 0 - 1
    known_clusters = 0
    paths = {}
    for cluster_number, subregion in enumerate(subregions):
        current_cluster = partition == cluster_number
        if subregion is None or levels_limit <= 1:
            result[current_cluster] = known_clusters
            paths[known_clusters] = (cluster_number,)
            known_clusters += 1
        else:
            local_partition, down_paths = _merged_partition(
                subregion.clustering.labels_, subregion.subregions, levels_limit - 1
            )
            result[current_cluster] = local_partition + known_clusters
            for cluster in np.unique(local_partition):
                paths[cluster + known_clusters] = (cluster_number, *down_paths[cluster])
            known_clusters += int(np.max(local_partition)) + 1
    return result, paths


def _update_graph(tree, size: int, graph: "networkx.Graph" = None, parent=None):
    tree_node = len(graph)
    graph.add_node(tree_node, size=size)
    if parent is not None:
        graph.add_edge(parent, tree_node)
    if tree is None:
        return
    for idx, subtree in enumerate(tree.subregions):
        _update_graph(
            subtree,
            size=np.sum(tree.clustering.labels_ == idx),
            graph=graph,
            parent=tree_node,
        )


def _as_graph(tree) -> "networkx.DiGraph":
    import networkx as nx

    graph = nx.DiGraph()
    _update_graph(tree, size=tree.partition.size, graph=graph, parent=None)
    return graph


def _make_labels(graph):
    import networkx as nx

    attributes = nx.get_node_attributes(graph, "size")
    return {idx: "{0} (size={1})".format(idx, attributes[idx]) for idx in graph}


def _make_sizes(graph):
    import networkx as nx

    values = np.array(list(nx.get_node_attributes(graph, "size").values()))
    return list(np.sqrt(values))


def scale_plot_size(factor=1.5):
    """Scale plot size in jupyter notebook."""
    import matplotlib as mpl

    default_dpi = mpl.rcParamsDefault["figure.dpi"]
    mpl.rcParams["figure.dpi"] = default_dpi * factor


def _make_positions(graph):
    import networkx as nx

    intermediate = nx.spectral_layout(graph)
    return nx.spring_layout(graph, pos=intermediate, iterations=20)


def plot(tree, with_size=False):
    """Plot visualization of splits."""
    graph = _as_graph(tree)
    arguments = {
        "G": graph,
        "pos": _make_positions(graph),
        "node_size": _make_sizes(graph),
        "font_size": 3,
        "width": 0.1,  # lines width
    }
    if with_size:
        arguments["labels"] = _make_labels(graph)
    import networkx as nx

    nx.draw_networkx(**arguments)
    import matplotlib.pyplot as plt

    plt.axis("off")


def reject_split(
    tree: Optional[u.DivikResult], rejection_size: int = 0
) -> Optional[u.DivikResult]:
    """Re-apply rejection condition on known result tree."""
    if tree is None:
        logging.debug("Rejecting empty.")
        return None
    counts = np.unique(tree.clustering.labels_, return_counts=True)[1]
    if any(counts <= rejection_size):
        logging.debug("Rejecting by condition.")
        return None
    allowed_subregions = [
        reject_split(subregion, rejection_size) for subregion in tree.subregions
    ]
    merged, _ = _merged_partition(tree.clustering.labels_, allowed_subregions)
    logging.debug("Returning pruned tree.")
    return u.DivikResult(
        clustering=tree.clustering,
        feature_selector=tree.feature_selector,
        merged=merged,
        subregions=allowed_subregions,
    )
