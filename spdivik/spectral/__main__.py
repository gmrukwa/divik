#!/usr/bin/env python
from functools import partial
import logging
import os
import pickle

import numpy as np
import skimage.io as sio

import spdivik.spectral as sp
import spdivik._scripting as scr
import spdivik.visualize as vis


def save(spectral: sp.AutoSpectralClustering, destination: str,
         xy: np.ndarray=None):
    logging.info("Saving result.")

    logging.info("Saving model.")
    fname = partial(os.path.join, destination)
    with open(fname('model.pkl'), 'wb') as pkl:
        pickle.dump(spectral, pkl)

    logging.info("Saving segmentation.")
    np.savetxt(fname('final_partition.csv'), spectral.labels_.reshape(-1, 1),
               delimiter=', ', fmt='%i')
    np.save(fname('final_partition.npy'), spectral.labels_.reshape(-1, 1))
    if xy is not None:
        visualization = vis.visualize(spectral.labels_, xy=xy)
        sio.imsave(fname('final_partition.png'), visualization)


def main():
    data, config, destination, xy = scr.initialize()
    try:
        spectral = sp.AutoSpectralClustering(**config)
        spectral.fit(data)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise
    save(spectral, destination, xy)


if __name__ == '__main__':
    main()
