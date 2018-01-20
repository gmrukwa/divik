import unittest
from unittest.mock import patch

import numpy as np

import spdivik.kmeans as km

import spdivik.distance as dist


data = np.array([
    [1, 1, 1, 1],
    [2, 4, 2, 2],
    [1.9, 4.2, 1.9, 1.9],
    [2, 2, 2, 2],
    [1.1, 0.8, 1.1, 1.1],
    [1000, 490231, -412342, -7012]
])
