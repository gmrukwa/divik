import json

from dash.dependencies import Output, Input, State

from divik._inspect import recolor as recolor, persistence as per
from divik._inspect.app import app
from divik._inspect.layout import Fields


@app.callback(
    Output(Fields.SELECTED_POINT, 'children'),
    [Input(Fields.CLUSTERS_GRAPH, 'clickData')]
)
def update_click_data(click_data):
    if click_data:
        point = click_data['points'][0]
        return json.dumps({
            'cluster': point['z'] if 'z' in point else 'outside the sample',
            'x': point['x'],
            'y': point['y']
        }, indent=4, sort_keys=True)
    return 'no point selected'


@app.callback(
    Output(Fields.CLUSTER_COLOR_SAMPLE, 'style'),
    [
        Input(Fields.CLUSTER_COLOR_R, 'value'),
        Input(Fields.CLUSTER_COLOR_G, 'value'),
        Input(Fields.CLUSTER_COLOR_B, 'value'),
    ]
)
def update_color_sample(r, g, b):
    return {
        'background-color': 'rgb({0}, {1}, {2})'.format(r, g, b),
        'height': '1em'
    }


@app.callback(
    Output(Fields.CLUSTER_COLOR_R, 'value'),
    [Input(Fields.SELECTED_POINT, 'children')],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def set_r_from_point(selected_point, figure):
    return recolor.as_rgb(selected_point, figure)[0]


@app.callback(
    Output(Fields.CLUSTER_COLOR_G, 'value'),
    [Input(Fields.SELECTED_POINT, 'children')],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def set_g_from_point(selected_point, figure):
    return recolor.as_rgb(selected_point, figure)[1]


@app.callback(
    Output(Fields.CLUSTER_COLOR_B, 'value'),
    [Input(Fields.SELECTED_POINT, 'children')],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def set_b_from_point(selected_point, figure):
    return recolor.as_rgb(selected_point, figure)[2]


@app.callback(
    Output(Fields.COLOR_OVERRIDES_STORAGE, 'children'),
    [
        Input(Fields.CLUSTER_COLOR_APPLY, 'n_clicks_timestamp'),
        Input(Fields.CLUSTER_COLOR_RESET, 'n_clicks_timestamp'),
        Input(Fields.LOAD_PROFILE, 'n_clicks_timestamp'),
    ],
    [
        State(Fields.COLOR_OVERRIDES_STORAGE, 'children'),
        State(Fields.CLUSTER_COLOR_SAMPLE, 'style'),
        State(Fields.LEVEL, 'value'),
        State(Fields.SELECTED_POINT, 'children'),
        State(Fields.SAVED_PROFILES, 'value')
    ]
)
def store_color_override(apply, reset, load, overrides, sample_style, level,
                         selected_point, name):
    apply = apply or 0
    reset = reset or 0
    load = load or 0
    last_click = max(apply, load, reset)
    if not last_click or reset == last_click:
        return '{}'
    if load == last_click:
        if name:
            return per.restore_color_overrides(name)
        return '{}'
    if apply == last_click:
        return recolor.update_color_overrides(
            overrides, sample_style['background-color'], level, selected_point)
    raise NotImplementedError('Unhandled case.')
