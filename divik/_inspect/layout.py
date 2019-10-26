from enum import Enum, unique

import dash_core_components as dcc
import dash_html_components as html

from divik._summary import depth
from divik._inspect.app import divik_result, divik_result_path
from divik._inspect.figure import default_clusters_figure


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
    'SAVED_PROFILES',
    'LOAD_PROFILE',
    'NEW_PROFILE_NAME',
    'SAVE_PROFILE',
    'MENU_TABS',
    'COLOR_MANIPULATION_TAB',
    'EXCLUSION_TAB',
    'PERSISTENCE_TAB',
]


# noinspection PyArgumentList
Fields = unique(Enum(
    'Fields', [(name, _as_id(name)) for name in _FIELDS], type=str))


def make_title():
    return html.H4(id=Fields.TITLE,
                   children=divik_result_path(),
                   style={'text-align': 'center'})


def make_plot():
    result_depth = depth(divik_result())
    return html.Div(id=Fields.CLUSTERS_CONTAINER, children=[
        dcc.Graph(id=Fields.CLUSTERS_GRAPH,
                  figure=default_clusters_figure(),
                  style={'min-height': 600}),
        html.H4('Level'),
        dcc.Slider(id=Fields.LEVEL, value=1, min=1, max=result_depth - 1,
                   step=1, marks={i: i for i in range(1, result_depth)})
    ], className='eight columns')


def make_selected_point_display():
    return html.Div([
        html.H4('Selected point'),
        html.Code(id=Fields.SELECTED_POINT)
    ])


def make_new_color_assignment_panel():
    return html.Div([
        html.H4('Assign new cluster color'),
        html.Div(id=Fields.CLUSTER_COLOR_SAMPLE),
        dcc.Slider(id=Fields.CLUSTER_COLOR_R, min=0, max=255, value=128),
        dcc.Slider(id=Fields.CLUSTER_COLOR_G, min=0, max=255, value=128),
        dcc.Slider(id=Fields.CLUSTER_COLOR_B, min=0, max=255, value=128),
        html.Button('Apply', id=Fields.CLUSTER_COLOR_APPLY),
    ])


def make_colors_reset_panel():
    return html.Div([
        html.H4('Reset clusters colors'),
        html.Button('Reset', id=Fields.CLUSTER_COLOR_RESET),
    ])


def make_disabled_clusters_panel():
    return html.Div([
        html.H4('Disabled clusters'),
        dcc.Dropdown(id=Fields.DISABLED_CLUSTERS_PICKER, multi=True)
    ])


def make_load_profile_panel():
    return html.Div([
        html.H4('Load profile'),
        dcc.Dropdown(id=Fields.SAVED_PROFILES),
        html.Br(),
        html.Button('Load', id=Fields.LOAD_PROFILE)
    ])


def make_save_profile_panel():
    return html.Div([
        html.H4('Save profile'),
        dcc.Input(id=Fields.NEW_PROFILE_NAME, type='text',
                  style={'display': 'block'}),
        html.Br(),
        html.Button('Save', id=Fields.SAVE_PROFILE)
    ])


def make_color_manipulation_tab():
    return html.Div(id=Fields.COLOR_MANIPULATION_TAB, children=[
        make_selected_point_display(),
        make_new_color_assignment_panel(),
        make_colors_reset_panel(),
    ])


def make_exclusion_tab():
    return html.Div(id=Fields.EXCLUSION_TAB, children=[
        make_disabled_clusters_panel(),
    ])


def make_persistence_tab():
    return html.Div(id=Fields.PERSISTENCE_TAB, children=[
        make_load_profile_panel(),
        make_save_profile_panel(),
    ])


def make_menu_panel():
    return html.Div(children=[
        dcc.Tabs(id=Fields.MENU_TABS,
                 value=Fields.COLOR_MANIPULATION_TAB,
                 children=[
                     dcc.Tab(label='COLOR',
                             value=Fields.COLOR_MANIPULATION_TAB),
                     dcc.Tab(label='SELECT',
                             value=Fields.EXCLUSION_TAB),
                     dcc.Tab(label='PROFILE',
                             value=Fields.PERSISTENCE_TAB),
                 ]),
        make_color_manipulation_tab(),
        make_exclusion_tab(),
        make_persistence_tab(),
    ], className='three columns')


def make_storage():
    return html.Div([
        html.Div(id=Fields.DISABLED_CLUSTERS_STORAGE, style={'display': 'none'}),
        html.Div(id=Fields.COLOR_OVERRIDES_STORAGE, style={'display': 'none'})
    ])


def make_layout():
    return html.Div([
        make_title(),
        make_plot(),
        make_menu_panel(),
        make_storage(),
    ])
