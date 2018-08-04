"""Conditions that allow to reject result of split."""

from typing import Callable

import numpy as np

from spdivik.types import ScoredSegmentation

RejectionCondition = Callable[[ScoredSegmentation], bool]


def reject_if_clusters_smaller_than(size: int,
                                    segmentation: ScoredSegmentation) -> bool:
    """Reject split if small clusters appear."""
    labels, _, _ = segmentation
    _, counts = np.unique(labels, return_counts=True)
    return np.any(counts < size)
