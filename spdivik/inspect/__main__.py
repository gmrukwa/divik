import argparse as agp

from spdivik.inspect.app import app
from spdivik.inspect.layout import LAYOUT


app.layout = LAYOUT


def parse_args():
    parser = agp.ArgumentParser()

    parser.add_argument('--result', '-r', help='divik result directory',
                        required=True)
    parser.add_argument('--xy', help='coordinates of points', required=True)
    parser.add_argument('--debug', help='enables debug mode', const=True,
                        default=False, action='store_const')

    return parser.parse_args()


def main():
    app.run_server(
        host='0.0.0.0',  # required for external access
        debug=True,
        dev_tools_hot_reload=True,
    )


if __name__ == '__main__':
    main()
