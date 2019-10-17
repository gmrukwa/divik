from dash.dependencies import Output, Input, State

from divik.inspect.app import app
from divik.inspect.layout import Fields


def make_tab_callback(tab_id):
    @app.callback(
        Output(tab_id, 'style'),
        [Input(Fields.MENU_TABS, 'value')],
        [State(tab_id, 'style')]
    )
    def update_visibility(value, style):
        style = style or {}
        if value == tab_id:
            style['display'] = 'block'
        else:
            style['display'] = 'none'
        return style


make_tab_callback(Fields.COLOR_MANIPULATION_TAB)
make_tab_callback(Fields.EXCLUSION_TAB)
make_tab_callback(Fields.PERSISTENCE_TAB)
