from copy import deepcopy

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


def _make_default_clusters_figure():
    current = deepcopy(_DEFAULT_CLUSTERS_FIGURE)
    current['data'][0]['x'] = xy().T[0]
    current['data'][0]['y'] = xy().T[1].max() - xy().T[1]
    return current


def _set_levels(level: int, current):
    partition = merged_partition(divik_result(), level)
    current['data'][0]['z'] = partition
    current['data'][0]['colorscale'] = make_colormap(partition)


def clusters_figure(level: int, title: str=None, current=None):
    if current is None:
        current = _make_default_clusters_figure()

    _set_levels(level, current)

    if title is not None:
        current['layout']['title'] = title

    return current
