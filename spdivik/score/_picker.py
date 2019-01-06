from abc import ABCMeta, abstractmethod
from multiprocessing.pool import Pool
from typing import List, Optional

import numpy as np

from spdivik.kmeans._core import KMeans
from spdivik.types import Data


class Picker(metaclass=ABCMeta):
    @abstractmethod
    def score(self, data: Data, estimators: List[KMeans], pool: Pool=None) \
            -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def select(self, scores: np.ndarray) -> Optional[int]:
        raise NotImplemented
