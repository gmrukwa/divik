from copy import deepcopy

import numpy as np

from spdivik.inspect.app import divik_result, xy
from spdivik.inspect.color import make_colormap
from spdivik.summary import merged_partition


_DEFAULT_CLUSTERS_FIGURE = {
    'data': [
        {
            'zsmooth': False,
            'type': 'heatmap',
            'colorscale': 'Rainbow',
            'showscale': False
        }
    ],
    'layout': {
        # 't': 40 is useful with title
        'margin': {'l': 0, 'b': 0, 'r': 0, 't': 0, 'pad': 0},
        # 'paper_bgcolor': '#7f7f7f',  # useful for debugging margins
        'autosize': True,
        'xaxis': {
            'showgrid': False,
            'showticklabels': False,
            'zeroline': False,
            'ticks': ''
        },
        'yaxis': {
            'showgrid': False,
            'showticklabels': False,
            'zeroline': False,
            'ticks': '',
            'scaleanchor': 'x',
            'scaleratio': 1
        }
    }
}


def default_clusters_figure():
    current = deepcopy(_DEFAULT_CLUSTERS_FIGURE)
    current['data'][0]['x'] = xy().T[0]
    current['data'][0]['y'] = xy().T[1].max() - xy().T[1]
    # customdata is used for filtering disabled clusters
    # True <=> cluster enabled, False <=> cluster disabled
    current['data'][0]['customdata'] = np.ones_like(xy().T[0])
    set_visualization_levels(1, current)
    return current


def get_enabled_ids(figure=None, enabled_flag=None, partition=None):
    if partition is None:
        assert figure is not None
        partition = figure['data'][0]['z']
    if enabled_flag is None:
        assert figure is not None
        enabled_flag = figure['data'][0]['customdata']
    return np.unique(partition[enabled_flag])


def get_disabled_ids(figure=None, enabled_flag=None, partition=None):
    if partition is None:
        assert figure is not None
        partition = figure['data'][0]['z']
    if enabled_flag is None:
        assert figure is not None
        enabled_flag = figure['data'][0]['customdata']
    enabled_ids = get_enabled_ids(enabled_flag=enabled_flag,
                                  partition=partition)
    return np.setdiff1d(partition, enabled_ids)


def get_all_labels(figure):
    return np.unique(figure['data'][0]['z'])


def vec_in(v1, v2):
    return np.max(v1.reshape(-1, 1) == v2.reshape(1, -1), axis=1)


def recompute_disabled(partition, old_enabled_flag):
    # if any subcluster is enabled, parent cluster will be enabled
    # if parent is disabled, children will be disabled
    new_enabled_id = get_enabled_ids(enabled_flag=old_enabled_flag,
                                     partition=partition)
    new_disabled_id = get_disabled_ids(enabled_flag=old_enabled_flag,
                                       partition=partition)
    new_enabled_flag = vec_in(partition, new_enabled_id)
    return new_enabled_flag, new_disabled_id


def set_visualization_levels(level: int, current):
    partition = merged_partition(divik_result(), level)
    enabled_flag, disabled_id = recompute_disabled(
        partition, current['data'][0]['customdata'] != 0)

    current['data'][0]['z'] = partition
    current['data'][0]['colorscale'] = make_colormap(partition,
                                                     disabled=disabled_id)
    current['data'][0]['customdata'] = enabled_flag

    return current
