from enum import Enum, unique

import dash_core_components as dcc
import dash_html_components as html

from spdivik.inspect.app import divik_result, xy
from spdivik.summary import merged_partition


def _as_id(name: str) -> str:
    return name.lower().replace('_', '-')


_FIELDS = ['TITLE', 'HEATMAP']


# noinspection PyArgumentList
Fields = unique(Enum(
    'Fields', [(name, _as_id(name)) for name in _FIELDS], type=str))


def make_layout():
    return html.Div([
        html.H1(id=Fields.TITLE, children='Visualization'),
        dcc.Graph(id=Fields.HEATMAP,
                  figure={
                      'data': [
                          {
                              'x': xy().T[0],
                              'y': xy().T[1].max() - xy().T[1],
                              'z': merged_partition(divik_result(), 1),
                              'zsmooth': False,
                              'type': 'heatmap',
                              'name': 'level 1',
                              'colorscale': 'Rainbow',
                              'showscale': False
                          }
                      ],
                      'layout': {
                          'title': 'Clusters',
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
                  })
    ])
