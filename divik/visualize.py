import argparse as agp

import numpy as np
from skimage.color import label2rgb
from skimage.io import imsave

from divik._data_io import load_data


def parse_args():
    parser = agp.ArgumentParser()

    parser.add_argument('--labels', required=True,
                        help='Path to file with labels')
    parser.add_argument('--xy', required=True,
                        help='Path to file with spatial coordinates')
    parser.add_argument('--destination', required=True,
                        help='Path to destination file')

    return parser.parse_args()


def visualize(label, xy, shape=None):
    x, y = xy.T
    if shape is None:
        shape = np.max(y) + 1, np.max(x) + 1
    y = y.max() - y
    label = label - label.min() + 1
    label_map = np.zeros(shape, dtype=int)
    label_map[y, x] = label
    image = label2rgb(label_map, bg_label=0)
    return image


def main():
    args = parse_args()
    print("Loading {0} as labels.".format(args.labels))
    print("Loading {0} as coordinates.".format(args.xy))
    print("Labels will be saved to {0}.".format(args.destination))
    xy = load_data(args.xy).astype(int)
    labels = load_data(args.labels).astype(int)
    visualization = visualize(label=labels, xy=xy)
    imsave(args.destination, visualization)


if __name__ == '__main__':
    main()
