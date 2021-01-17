import logging as lg
from typing import List

import numpy as np
import tqdm


def _constant_rows(matrix: np.ndarray) -> List[int]:
    is_constant = matrix.min(axis=1) == matrix.max(axis=1)
    return np.where(is_constant)[0]


class DivikReporter:
    def __init__(self, progress_reporter: tqdm.tqdm = None, warn_const: bool = True):
        self.progress_reporter = progress_reporter
        self.paths_open = 1
        self.warn_const = warn_const

    def filter(self, subset):
        lg.info("Feature filtering.")
        if lg.getLogger().getEffectiveLevel() <= lg.DEBUG:
            lg.debug("Subset shape: {0}".format(subset.shape))
            lg.debug("Has NaNs: {0}".format(np.isnan(subset).any()))
            lg.debug("Limits: min={0}; max={1}".format(subset.min(), subset.max()))
            lg.debug("Has constant rows: {0}".format(_constant_rows(subset)))

    def filtered(self, data):
        lg.debug("Shape after filtering: {0}".format(data.shape))
        constant = _constant_rows(data)
        if self.warn_const and any(constant):
            msg = (
                "After feature filtering some rows are constant: {0}. "
                "This may not work with specific configurations."
            )
            lg.warning(msg.format(constant))

    def stop_check(self):
        lg.info("Stop condition check.")

    def finished_for(self, n_observations: int):
        self.paths_open -= 1
        lg.info(
            "Stop condition fired for {0}. {1} paths open.".format(
                n_observations, self.paths_open
            )
        )
        if self.progress_reporter is not None:
            self.progress_reporter.update(n_observations)

    def rejected(self, n_observations: int):
        self.paths_open -= 1
        lg.info(
            "Rejected segmentation of {0}. {1} paths open.".format(
                n_observations, self.paths_open
            )
        )
        if self.progress_reporter is not None:
            self.progress_reporter.update(n_observations)

    def processing(self, dataset: np.ndarray):
        lg.info(
            "Processing subset with {0} observations and {1} features.".format(
                *dataset.shape
            )
        )

    def recurring(self, n_subregions):
        self.paths_open += n_subregions
        lg.info(
            "Recurring into {0} subregions. {1} paths open.".format(
                n_subregions, self.paths_open
            )
        )

    def assemble(self):
        self.paths_open -= 1
        lg.info("Assembled. {0} paths open.".format(self.paths_open))
