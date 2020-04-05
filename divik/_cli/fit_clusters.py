import logging

import gin

from divik.core import dump_gin_args, parse_gin_args
from ._model_io import save
from ._utils import (
    prepare_destination,
    setup_logger,
    try_load_data,
    try_load_xy,
)


@gin.configurable
def load_data(path=gin.REQUIRED):
    return try_load_data(path)


@gin.configurable
def load_xy(path=None):
    return try_load_xy(path)


@gin.configurable
def experiment(
    model=gin.REQUIRED,
    destination: str = 'result',
    omit_datetime: bool = False,
    verbose: bool = False,
    exist_ok: bool = False,
):
    destination = prepare_destination(
        destination, omit_datetime, exist_ok=exist_ok)
    dump_gin_args(destination)
    setup_logger(destination, verbose)
    logging.info(str(model))
    data = load_data()
    xy = load_xy()
    # repeated dump just because the dataset locations are not tracked
    dump_gin_args(destination)
    model.fit(data)
    save(model, destination, xy=xy)


def main():
    parse_gin_args()
    experiment()


if __name__ == '__main__':
    main()
