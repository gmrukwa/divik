from dash.dependencies import Output, Input, State

from divik._inspect import persistence as per
from divik._inspect.app import app
from divik._inspect.layout import Fields


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
