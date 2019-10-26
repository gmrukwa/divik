import json

from dash.dependencies import Output, Input, State

from divik._inspect import figure as fig
from divik._inspect.app import app
from divik._inspect.layout import Fields


@app.callback(
    Output(Fields.CLUSTERS_GRAPH, 'figure'),
    [
        Input(Fields.LEVEL, 'value'),
        Input(Fields.DISABLED_CLUSTERS_STORAGE, 'children'),
        Input(Fields.COLOR_OVERRIDES_STORAGE, 'children'),
    ],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def update_visualization(depth, disabled_state, overrides, figure):
    disabled = json.loads(disabled_state)['excluded'] if disabled_state else []
    overrides = json.loads(overrides).get(str(depth), {}) if overrides else {}
    return fig.update_clusters_figure(depth, disabled, overrides, figure)
