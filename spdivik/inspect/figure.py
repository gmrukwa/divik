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


def set_visualization_levels(level: int, current):
    partition = merged_partition(divik_result(), level)

    # if any subcluster is enabled, parent cluster will be enabled
    # if parent is disabled, children will be disabled
    old_enabled_flag = current['data'][0]['customdata'] != 0
    new_enabled_id = np.unique(partition[old_enabled_flag])
    new_disabled_id = np.setdiff1d(partition, new_enabled_id)
    new_enabled_flag = np.max(
        partition.reshape(-1, 1) == new_enabled_id.reshape(1, -1), axis=1)

    current['data'][0]['z'] = partition
    current['data'][0]['colorscale'] = make_colormap(partition,
                                                     disabled=new_disabled_id)
    current['data'][0]['customdata'] = new_enabled_flag

    return current
