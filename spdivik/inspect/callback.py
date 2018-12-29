from dash.dependencies import Input, Output

from spdivik.inspect.app import app
from spdivik.inspect.figure import clusters_figure
from spdivik.inspect.layout import Fields


@app.callback(
    Output(Fields.CLUSTERS, 'figure'),
    [Input(Fields.LEVEL, 'value')]
)
def update_depth(depth):
    return clusters_figure(depth, 'Clusters')
