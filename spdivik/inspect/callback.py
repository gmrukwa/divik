import json

from dash.dependencies import Input, Output, State

from spdivik.inspect.app import app
import spdivik.inspect.recolor as recolor
import spdivik.inspect.exclusion as ex
import spdivik.inspect.figure as fig
from spdivik.inspect.layout import Fields


@app.callback(
    Output(Fields.CLUSTERS_GRAPH, 'figure'),
    [
        Input(Fields.LEVEL, 'value'),
        Input(Fields.DISABLED_CLUSTERS_STORAGE, 'children')
    ],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def update_visualization(depth, disabled_state, current_figure):
    disabled = json.loads(disabled_state)['excluded'] if disabled_state else []
    return fig.update_clusters_figure(depth, disabled, current_figure)


@app.callback(
    Output(Fields.DISABLED_CLUSTERS_STORAGE, 'children'),
    [Input(Fields.DISABLED_CLUSTERS_PICKER, 'value')],
    [State(Fields.LEVEL, 'value'), State(Fields.DISABLED_CLUSTERS_STORAGE, 'children')]
)
def update_disabled_clusters_to_new_level(disabled_clusters, level, old_state):
    if not old_state:
        return ex.initialize_storage(level)
    return ex.update_storage(level, disabled_clusters, old_state)


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
    else:
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
