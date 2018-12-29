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
    'DISABLED_CLUSTERS_STORAGE',
    'SELECTED_POINT',
    'COLOR_OVERRIDES_STORAGE',
    'CLUSTER_COLOR_R',
    'CLUSTER_COLOR_G',
    'CLUSTER_COLOR_B',
    'CLUSTER_COLOR_SAMPLE',
    'CLUSTER_COLOR_APPLY',
    'CLUSTER_COLOR_RESET',
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
            html.Div([
                html.H4('Selected point'),
                html.Code(id=Fields.SELECTED_POINT)
            ]),

            html.Div([
                html.H4('Assign new cluster color'),
                html.Div(id=Fields.CLUSTER_COLOR_SAMPLE),
                dcc.Slider(id=Fields.CLUSTER_COLOR_R, min=0, max=255, value=128),
                dcc.Slider(id=Fields.CLUSTER_COLOR_G, min=0, max=255, value=128),
                dcc.Slider(id=Fields.CLUSTER_COLOR_B, min=0, max=255, value=128),
                html.Button('Apply', id=Fields.CLUSTER_COLOR_APPLY),
            ]),

            html.Div([
                html.H4('Reset clusters colors'),
                html.Button('Reset', id=Fields.CLUSTER_COLOR_RESET),
            ]),

            html.Div([
                html.H4('Disabled clusters'),
                dcc.Dropdown(id=Fields.DISABLED_CLUSTERS_PICKER,
                             multi=True)
            ])
        ], className='three columns'),

        html.Div(id=Fields.DISABLED_CLUSTERS_STORAGE, style={'display': 'none'}),
        html.Div(id=Fields.COLOR_OVERRIDES_STORAGE, style={'display': 'none'})
    ])
