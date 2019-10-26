import unittest
from unittest.mock import MagicMock

from typing import List

import numpy as np

import divik._score


def simulate_values(values: List):
    return MagicMock(side_effect=values)


def use(func):
    return MagicMock(side_effect=func)


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        dummy_scores = [1, 3, 2]
        self.score = simulate_values(dummy_scores)
        self.segmentation_method = use(lambda data, k: (data, k))
        self.parameters = [divik._score.ParameterValues(name='k',
                                                        values=[4, 5, 6])]
        self.best_parameter_value = 5
        self.optimizer = divik._score.Optimizer(
            self.score, self.segmentation_method, self.parameters)
        self.data = np.array([[1], [2], [3]])

    def test_initializes(self):
        pass

    def test_iterates_parameter_values(self):
        self.optimizer(self.data)
        self.segmentation_method.assert_any_call(self.data, k=4)
        self.segmentation_method.assert_any_call(self.data, k=5)
        self.segmentation_method.assert_any_call(self.data, k=6)

    def test_finds_best_solution(self):
        result = self.optimizer(self.data)
        expected = self.segmentation_method(self.data,
                                            k=self.best_parameter_value)
        self.assertSequenceEqual(result[:2], expected)
        unexpected = self.segmentation_method(self.data,
                                              k=self.best_parameter_value - 1)
        assert any(left is right for left, right in zip(expected, unexpected))
