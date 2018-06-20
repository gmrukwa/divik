import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def depth(tree, children_collection_name='subregions'):
    if tree is None:
        return 1
    return max(depth(subtree) for subtree in getattr(tree, children_collection_name)) + 1


def total_number_of_clusters(tree) -> int:
    if tree is None:
        return 1
    return sum(total_number_of_clusters(subtree) for subtree in tree.subregions)


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
