"""Conditions that allow to reject result of split."""

from typing import Callable

import numpy as np

from divik.types import ScoredSegmentation

RejectionCondition = Callable[[ScoredSegmentation], bool]


def reject_if_clusters_smaller_than(segmentation: ScoredSegmentation,
                                    size: int = None,
                                    percentage: float = None) -> bool:
    """Reject split if small clusters appear."""
    assert size is not None or percentage is not None
    assert percentage is None or (0. <= percentage <= 1.)
    labels, _, _ = segmentation
    _, counts = np.unique(labels, return_counts=True)
    if size is None:
        size = percentage * labels.size
    return np.any(counts < size)
