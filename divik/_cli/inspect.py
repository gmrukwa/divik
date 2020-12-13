import argparse as agp
import glob
import logging
import os
from itertools import chain
from typing import List

from divik.core.io import DIVIK_RESULT_FNAME
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


def _result_path_patterns(slug: str) -> List[str]:
    slug_pattern = '*{0}*'.format(slug)
    direct = os.path.join(slug_pattern, DIVIK_RESULT_FNAME)
    prefixed = os.path.join('**', slug_pattern, DIVIK_RESULT_FNAME)
    suffixed = os.path.join(slug_pattern, '**', DIVIK_RESULT_FNAME)
    bothfixed = os.path.join('**', slug_pattern, '**', DIVIK_RESULT_FNAME)
    return list((direct, prefixed, suffixed, bothfixed))


def _find_possible_directories(patterns: List[str]) -> List[str]:
    possible_locations = chain.from_iterable(
        glob.glob(pattern, recursive=True) for pattern in patterns)
    possible_paths = list({
        os.path.split(fname)[0] for fname in possible_locations
    })
    return possible_paths


def as_divik_result_path(path_or_slug: str):
    possible_location = os.path.join(path_or_slug, DIVIK_RESULT_FNAME)
    if os.path.exists(possible_location):
        return path_or_slug
    patterns = _result_path_patterns(path_or_slug)
    possible_paths = _find_possible_directories(patterns)
    if not possible_paths:
        raise FileNotFoundError(path_or_slug)
    if len(possible_paths) > 1:
        msg = 'Multiple possible result directories: {0}. Selecting {1}.'
        logging.warning(msg.format(possible_paths, possible_paths[0]))
    return possible_paths[0]


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
