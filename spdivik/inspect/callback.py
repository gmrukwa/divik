from dash.dependencies import Input, Output, State

from spdivik.inspect.app import app
from spdivik.inspect.figure import set_visualization_levels
from spdivik.inspect.layout import Fields


@app.callback(
    Output(Fields.CLUSTERS_GRAPH, 'figure'),
    [Input(Fields.LEVEL, 'value')],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def update_visualization_depth(depth, current_figure):
    return set_visualization_levels(depth, current_figure)
