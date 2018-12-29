from dash.dependencies import Input, Output, State

from spdivik.inspect.app import app
import spdivik.inspect.exclusion as ex
import spdivik.inspect.figure as fig
from spdivik.inspect.layout import Fields


@app.callback(
    Output(Fields.CLUSTERS_GRAPH, 'figure'),
    [Input(Fields.LEVEL, 'value')],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def update_visualization_depth(depth, current_figure):
    return fig.set_visualization_levels(depth, current_figure)


@app.callback(
    Output(Fields.ENABLED_CLUSTERS_PICKER, 'options'),
    [Input(Fields.LEVEL, 'value')],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def update_possible_enabled_clusters(_, current_figure):
    return ex.as_dropdown_options(fig.get_all_labels(current_figure))


@app.callback(
    Output(Fields.DISABLED_CLUSTERS_PICKER, 'options'),
    [Input(Fields.LEVEL, 'value')],
    [State(Fields.CLUSTERS_GRAPH, 'figure')]
)
def update_possible_enabled_clusters(_, current_figure):
    return ex.as_dropdown_options(fig.get_all_labels(current_figure))
