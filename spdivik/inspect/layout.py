from enum import Enum, unique

import dash_core_components as dcc
import dash_html_components as html

from spdivik.inspect.figure import clusters_figure


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
                  figure=clusters_figure(1, 'Clusters'))
    ])
