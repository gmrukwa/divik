from spdivik.inspect.app import divik_result, xy
from spdivik.summary import merged_partition


def clusters_figure(level: int, title: str):
    return {
        'data': [
            {
                'x': xy().T[0],
                'y': xy().T[1].max() - xy().T[1],
                'z': merged_partition(divik_result(), level),
                'zsmooth': False,
                'type': 'heatmap',
                'colorscale': 'Rainbow',
                'showscale': False
            }
        ],
        'layout': {
            'title': title,
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
