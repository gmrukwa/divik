from abc import ABCMeta, abstractmethod
from multiprocessing.pool import Pool
from typing import List, Optional

import numpy as np
import pandas as pd

from divik.types import Data


KMeans = 'divik.kmeans._core.KMeans'


class Picker(metaclass=ABCMeta):
    @abstractmethod
    def score(self, data: Data, estimators: List[KMeans], pool: Pool = None) \
            -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def select(self, scores: np.ndarray) -> Optional[int]:
        raise NotImplementedError

    @abstractmethod
    def report(self, estimators: List[KMeans], scores: np.ndarray) \
            -> pd.DataFrame:
        raise NotImplementedError
