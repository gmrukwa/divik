import argparse as agp

from skimage.io import imsave

from divik._cli._data_io import load_data
from divik.core import visualize


def parse_args():
    parser = agp.ArgumentParser()

    parser.add_argument('--labels', required=True,
                        help='Path to file with labels')
    parser.add_argument('--xy', required=True,
                        help='Path to file with spatial coordinates')
    parser.add_argument('--destination', required=True,
                        help='Path to destination file')

    return parser.parse_args()


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
