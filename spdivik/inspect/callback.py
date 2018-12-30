import json

from dash.dependencies import Input, Output, State

from spdivik.inspect.app import app
import spdivik.inspect.recolor as recolor
import spdivik.inspect.exclusion as ex
import spdivik.inspect.figure as fig
from spdivik.inspect.layout import Fields
import spdivik.inspect.persistence as per


@app.callback(
    Output(Fields.CLUSTERS_GRAPH, 'figure'),
    [
        Input(Fields.LEVEL, 'value'),
        Input(Fields.DISABLED_CLUSTERS_STORAGE, 'children'),
        Input(Fields.COLOR_OVERRIDES_STORAGE, 'children'),
    ],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def update_visualization(depth, disabled_state, color_overrides, current_figure):
    disabled = json.loads(disabled_state)['excluded'] if disabled_state else []
    overrides = json.loads(color_overrides).get(str(depth), {}) if color_overrides else {}
    return fig.update_clusters_figure(depth, disabled, overrides, current_figure)


@app.callback(
    Output(Fields.DISABLED_CLUSTERS_STORAGE, 'children'),
    [
        Input(Fields.DISABLED_CLUSTERS_PICKER, 'value'),
        Input(Fields.LOAD_PROFILE, 'n_clicks'),
    ],
    [
        State(Fields.LEVEL, 'value'),
        State(Fields.DISABLED_CLUSTERS_STORAGE, 'children'),
        State(Fields.SAVED_PROFILES, 'value'),
    ]
)
def update_disabled_clusters_to_new_level(disabled_clusters, _, level,
                                          old_state, name):
    if not old_state:
        return ex.initialize_storage(level)
    if ex.got_update(level, disabled_clusters, old_state):
        return ex.update_storage(level, disabled_clusters, old_state)
    if name:
        return per.restore_disabled_clusters(name)
    return ex.initialize_storage(level)


@app.callback(
    Output(Fields.DISABLED_CLUSTERS_PICKER, 'options'),
    [Input(Fields.LEVEL, 'value')]
)
def update_possible_enabled_clusters(level):
    return ex.get_options(level)


@app.callback(
    Output(Fields.DISABLED_CLUSTERS_PICKER, 'value'),
    [Input(Fields.LEVEL, 'value')],
    [State(Fields.DISABLED_CLUSTERS_STORAGE, 'children')]
)
def update_actually_disabled_clusters(level, storage):
    return ex.update_actual_clusters(level, storage)


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
def set_r_from_point(selected_point, figure):
    return recolor.as_rgb(selected_point, figure)[1]


@app.callback(
    Output(Fields.CLUSTER_COLOR_B, 'value'),
    [Input(Fields.SELECTED_POINT, 'children')],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def set_r_from_point(selected_point, figure):
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


@app.callback(
    Output(Fields.SAVED_PROFILES, 'options'),
    [Input(Fields.SAVE_PROFILE, 'n_clicks')],
    [
        State(Fields.NEW_PROFILE_NAME, 'value'),
        State(Fields.DISABLED_CLUSTERS_STORAGE, 'children'),
        State(Fields.COLOR_OVERRIDES_STORAGE, 'children'),
        State(Fields.LEVEL, 'value'),
    ]
)
def save_profile(_, name, disabled, color, level):
    if name:
        per.preserve_by_name(name, disabled, color, level)
    return per.find_preserved()


@app.callback(
    Output(Fields.LEVEL, 'value'),
    [Input(Fields.LOAD_PROFILE, 'n_clicks')],
    [
        State(Fields.SAVED_PROFILES, 'value'),
        State(Fields.LEVEL, 'value')
    ]
)
def load_level_from_profile(_, name, self):
    if not name:
        return self
    return per.restore_level(name)
