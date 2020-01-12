#!/usr/bin/env python
import logging

from divik.cluster import DunnDiviK
import divik._cli._utils as sc

from divik._cli.divik import save


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
