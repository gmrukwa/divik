from copy import deepcopy

from divik._inspect.app import divik_result, xy
from divik._inspect.color import make_colormap
from divik._summary import merged_partition


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
    update_clusters_figure(1, [], {}, current)
    return current


def update_clusters_figure(level: int, disabled, color_overrides, current):
    partition = merged_partition(divik_result(), level)

    current['data'][0]['z'] = partition
    current['data'][0]['colorscale'] = make_colormap(partition,
                                                     disabled=disabled,
                                                     overrides=color_overrides)

    return current
