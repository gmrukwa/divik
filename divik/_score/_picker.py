from abc import ABCMeta, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd

from divik._utils import Data

KMeans = 'divik.KMeans'


class Picker(metaclass=ABCMeta):
    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    @abstractmethod
    def score(self, data: Data, estimators: List[KMeans]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def select(self, scores: np.ndarray) -> Optional[int]:
        raise NotImplementedError

    @abstractmethod
    def report(self, estimators: List[KMeans], scores: np.ndarray) \
            -> pd.DataFrame:
        raise NotImplementedError
