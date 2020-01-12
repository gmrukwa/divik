#!/usr/bin/env python
import json
import logging
import os
import pickle
import typing
import numpy as np
import pandas as pd
import skimage.io as sio

from divik._cli._data_io import DIVIK_RESULT_FNAME
from divik.cluster import DunnDiviK
import divik._summary as _smr
import divik._cli._utils as sc
import divik._utils as u

from divik._cli.divik import (
    _make_summary,
    _make_merged,
    _save_merged,
    _save
)


def main():
    data, config, destination, xy = sc.initialize()
    logging.info('Workspace initialized.')
    logging.info('Scenario configuration: {0}'.format(config))
    divik = DunnDiviK(**config)
    logging.info("Launching experiment.")
    try:
        divik.fit(data)
    except Exception as ex:
        logging.error("Failed with exception.")
        logging.error(repr(ex))
        raise
    save(data, divik, destination, xy)


if __name__ == '__main__':
    main()
