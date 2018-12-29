from dash.dependencies import Input, Output, State

from spdivik.inspect.app import app
from spdivik.inspect.figure import clusters_figure
from spdivik.inspect.layout import Fields


@app.callback(
    Output(Fields.CLUSTERS_GRAPH, 'figure'),
    [Input(Fields.LEVEL, 'value')],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def update_depth(depth, current_figure):
    return clusters_figure(depth, current=current_figure)
