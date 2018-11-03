import argparse as agp

import numpy as np
from skimage.color import label2rgb
from skimage.io import imsave

from spdivik._scripting import load_data


def parse_args():
    parser = agp.ArgumentParser()

    parser.add_argument('--labels', required=True,
                        help='Path to file with labels')
    parser.add_argument('--xy', required=True,
                        help='Path to file with spatial coordinates')
    parser.add_argument('--destination', required=True,
                        help='Path to destination file')

    return parser.parse_args


def visualize(label, x, y, shape):
    y = y.max() - y
    label = label - label.min() + 1
    label_map = np.zeros(shape, dtype=int)
    label_map[y, x] = label
    image = label2rgb(label_map, bg_label=0)
    return image


def main():
    args = parse_args()
    xy = load_data(args.xy)
    labels = load_data(args.labels)
    x, y = xy.T
    shape = np.max(y) + 1, np.max(x) + 1
    visualization = visualize(labels, x=x, y=y, shape=shape)
    imsave(args.destination, visualization)


if __name__ == '__main__':
    main()
