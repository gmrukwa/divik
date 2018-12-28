import dash_html_components as html

from spdivik.inspect.app import app
from spdivik.inspect.layout import LAYOUT


app.layout = html.Div(LAYOUT)


def main():
    app.run_server(
        host='0.0.0.0',  # required for external access
        debug=True,
        dev_tools_hot_reload=True,
    )


if __name__ == '__main__':
    main()
