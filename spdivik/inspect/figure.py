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


def _make_default_clusters_figure():
    current = deepcopy(_DEFAULT_CLUSTERS_FIGURE)
    current['data'][0]['x'] = xy().T[0]
    current['data'][0]['y'] = xy().T[1].max() - xy().T[1]
    return current


def _set_levels(level: int, current, disabled_clusters=None):
    partition = merged_partition(divik_result(), level)
    current['data'][0]['z'] = partition
    current['data'][0]['colorscale'] = make_colormap(partition,
                                                     disabled=disabled_clusters)


def clusters_figure(level: int, title: str=None, current=None,
                    disabled_clusters=None):
    if current is None:
        current = _make_default_clusters_figure()

    _set_levels(level, current, disabled_clusters=disabled_clusters)

    if title is not None:
        current['layout']['title'] = title

    return current
