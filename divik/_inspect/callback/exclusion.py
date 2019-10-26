from dash.dependencies import Output, Input, State

from divik._inspect import exclusion as ex, persistence as per
from divik._inspect.app import app
from divik._inspect.layout import Fields


@app.callback(
    Output(Fields.DISABLED_CLUSTERS_STORAGE, 'children'),
    [
        Input(Fields.DISABLED_CLUSTERS_PICKER, 'value'),
    ],
    [
        State(Fields.LOAD_PROFILE, 'n_clicks_timestamp'),
        State(Fields.LEVEL, 'value'),
        State(Fields.DISABLED_CLUSTERS_STORAGE, 'children'),
    ]
)
def update_disabled_clusters_to_new_level(disabled_clusters, stamp, level,
                                          old_state):
    if not old_state:
        return ex.initialize_storage(level)
    return ex.update_storage(level, disabled_clusters, stamp, old_state)


@app.callback(
    Output(Fields.DISABLED_CLUSTERS_PICKER, 'options'),
    [Input(Fields.LEVEL, 'value')]
)
def update_possible_enabled_clusters(level):
    return ex.get_options(level)


@app.callback(
    Output(Fields.DISABLED_CLUSTERS_PICKER, 'value'),
    [Input(Fields.LEVEL, 'value')],
    [
        State(Fields.DISABLED_CLUSTERS_STORAGE, 'children'),
        State(Fields.LOAD_PROFILE, 'n_clicks_timestamp'),
        State(Fields.SAVED_PROFILES, 'value'),
    ]
)
def update_actually_disabled_clusters(level, storage, stamp, name):
    if not storage:
        return []
    if ex.is_reloaded(stamp, storage):
        return per.restore_disabled_clusters(name)
    return ex.update_actual_clusters(level, storage)
