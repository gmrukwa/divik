from enum import Enum, unique

import dash_core_components as dcc
import dash_html_components as html


def _as_id(name: str) -> str:
    return name.lower().replace('_', '-')


_FIELDS = ['TITLE']


# noinspection PyArgumentList
Fields = unique(Enum(
    'Fields', [(name, _as_id(name)) for name in _FIELDS], type=str))


LAYOUT = html.P(id=Fields.TITLE, children='Hello World!')
