from enum import Enum, unique

import dash_core_components as dcc
import dash_html_components as html

from spdivik.summary import depth
from spdivik.inspect.app import divik_result, divik_result_path
from spdivik.inspect.figure import default_clusters_figure


def _as_id(name: str) -> str:
    return name.lower().replace('_', '-')


_FIELDS = [
    'TITLE',
    'CLUSTERS_CONTAINER',
    'CLUSTERS_GRAPH',
    'LEVEL',
    'DISABLED_CLUSTERS_PICKER',
    'DISABLED_CLUSTERS_STORAGE'
]


# noinspection PyArgumentList
Fields = unique(Enum(
    'Fields', [(name, _as_id(name)) for name in _FIELDS], type=str))


def make_layout():
    result_depth = depth(divik_result())
    return html.Div([
        html.H4(id=Fields.TITLE, children=divik_result_path(), style={'text-align': 'center'}),

        html.Div(id=Fields.CLUSTERS_CONTAINER, children=[
            dcc.Graph(id=Fields.CLUSTERS_GRAPH,
                      figure=default_clusters_figure(),
                      style={'min-height': 600}),
            html.H4('Level'),
            dcc.Slider(id=Fields.LEVEL,
                       value=1,
                       min=1,
                       max=result_depth - 1,
                       step=1,
                       marks={i: i for i in range(1, result_depth)})
        ], className='eight columns'),

        html.Div(children=[
            html.P('Color picker placeholder', style={'background-color': 'red'}),
            html.Div([
                html.H4('Disabled clusters'),
                dcc.Dropdown(id=Fields.DISABLED_CLUSTERS_PICKER,
                             multi=True)
            ])
        ], className='three columns'),

        html.Div(id=Fields.DISABLED_CLUSTERS_STORAGE, style={'display': 'none'})
    ])
