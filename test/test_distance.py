"""
test_distance.py
Tests distance module.

Copyright 2018 Spectre Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from unittest.mock import patch

import numpy as np

import spdivik.distance as dist


class IntradistanceCall(NotImplementedError):
    pass


class InterdistanceCall(NotImplementedError):
    pass


class DummyDistanceMetric(dist.DistanceMetric):
    def _intradistance(self, *_):
        raise IntradistanceCall(self._intradistance.__name__)

    def _interdistance(self, *_):
        raise InterdistanceCall(self._interdistance.__name__)


# noinspection PyCallingNonCallable
class DistanceMetricTest(unittest.TestCase):
    def setUp(self):
        self.metric = DummyDistanceMetric()
        self.first = np.array([[1], [2], [3]])
        self.second = np.array([[4], [5]])

    def test_throws_when_input_is_not_an_array(self):
        with self.assertRaises(ValueError):
            self.metric([[1]], np.array([[1]]))
        with self.assertRaises(ValueError):
            self.metric(np.array([[1]]), [[1]])

    def test_throws_when_input_is_not_2D(self):
        with self.assertRaises(ValueError):
            self.metric(np.array([1]), np.array([[1]]))
        with self.assertRaises(ValueError):
            self.metric(np.array([[1]]), np.array([1]))

    def test_calls_intradistance_for_the_same_array(self):
        array = np.array([[1]])
        with self.assertRaises(IntradistanceCall):
            self.metric(array, array)

    def test_calls_interdistance_for_different_arrays(self):
        with self.assertRaises(InterdistanceCall):
            self.metric(np.array([[1]]), np.array([[2]]))

    def test_checks_output_type_debug(self):
        with patch.object(self.metric, '_interdistance') as mock_distance:
            mock_distance.return_value = [[0]]
            with self.assertRaises(AssertionError):
                self.metric(self.first, self.second)

    def test_checks_output_dimension_in_debug(self):
        with patch.object(self.metric, '_interdistance') as mock_distance:
            mock_distance.return_value = np.array([0])
            with self.assertRaises(AssertionError):
                self.metric(self.first, self.second)

    def test_checks_output_size_in_debug(self):
        with patch.object(self.metric, '_interdistance') as mock_distance:
            mock_distance.return_value = np.array([[0]])
            with self.assertRaises(AssertionError):
                self.metric(self.first, self.second)
            mock_distance.return_value = np.array([[0], [0], [0]])
            with self.assertRaises(AssertionError):
                self.metric(self.first, self.second)
            mock_distance.return_value = np.array([[0, 0], [0, 0], [0, 0]])
            self.metric(self.first, self.second)
