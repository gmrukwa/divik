import unittest

import numpy as np
import numpy.testing as npt
from sklearn.base import clone

from divik.sampler._core import BaseSampler, ParallelSampler


class DummySampler(BaseSampler):
    def __init__(self, whatever):
        self.whatever = whatever

    def get_sample(self, seed):
        return np.array([[seed]])


class BaseSamplerTest(unittest.TestCase):
    def test_iterates_through_samples(self):
        sampler = DummySampler(1)
        samples = [v for _, v in zip(range(10000), sampler)]
        self.assertEqual(10000, len(samples))
        npt.assert_array_equal(np.ravel(samples), np.ravel(range(10000)))

    def test_fit_returns_self(self):
        sampler = DummySampler(1)
        self.assertIs(sampler, sampler.fit(np.array([[1]])))

    def test_parallel_sampler_generates_same_values(self):
        sampler = DummySampler(1)
        expected = sampler.get_sample(34134123)
        with sampler.parallel() as sampler_:
            actual = sampler_.get_sample(34134123)
            npt.assert_array_equal(expected, actual)

    def test_context_creates_parallel_sampler(self):
        sampler = DummySampler(1)
        with sampler.parallel() as sampler_:
            self.assertIsInstance(sampler_, ParallelSampler)

    def test_parallel_sampler_is_cloneable(self):
        sampler = DummySampler(1)
        with sampler.parallel() as sampler_:
            cloned = sampler_.clone()
            self.assertIsInstance(cloned, DummySampler)
            self.assertEqual(sampler.whatever, cloned.whatever)
    
    def test_parallel_sampler_is_not_iterable(self):
        sampler = DummySampler(1)
        with sampler.parallel() as sampler_, self.assertRaises(TypeError):
            iter(sampler_)
    
    def test_parallel_sampler_is_not_fittable(self):
        sampler = DummySampler(1)
        with sampler.parallel() as sampler_, self.assertRaises(AttributeError):
            sampler_.fit()

    def test_is_cloneable(self):
        original = DummySampler(1)
        cloned = clone(original)
        self.assertEqual(cloned.whatever, 1)
