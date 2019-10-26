import argparse as agp

from divik._cli._data_io import as_divik_result_path
from divik._inspect.app import app, divik_result, xy
# noinspection PyUnresolvedReferences
import divik._inspect.callback
from divik._inspect.layout import make_layout


def parse_args():
    parser = agp.ArgumentParser()

    parser.add_argument('--result', '-r', help='divik result directory',
                        required=True)
    parser.add_argument('--xy', help='coordinates of points', required=True)

    parser.add_argument('--host', default='127.0.0.1',
                        help='Sets up a host interface to run the visualization'
                             ' on. Use 0.0.0.0 while in Docker.')
    parser.add_argument('--debug', help='enables debug mode', const=True,
                        default=False, action='store_const')

    return parser.parse_args()


def main():
    args = parse_args()
    path = as_divik_result_path(args.result)
    divik_result(path)
    xy(args.xy)
    app.layout = make_layout()
    app.run_server(
        host=args.host,  # required for external access
        debug=args.debug,
        dev_tools_hot_reload=True,
    )


if __name__ == '__main__':
    main()
