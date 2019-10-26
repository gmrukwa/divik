import unittest
from unittest.mock import patch
import os

import divik._utils as u


@patch.object(os, os.cpu_count.__name__, new=lambda: 4)
class NJobsTest(unittest.TestCase):
    def test_gets_n_for_n_lower_than_cpus(self):
        self.assertEqual(u.get_n_jobs(3), 3)

    def test_gets_n_for_n_equal_cpus(self):
        self.assertEqual(u.get_n_jobs(4), 4)

    def test_gets_all_for_minus_one(self):
        self.assertEqual(u.get_n_jobs(-1), 4)

    def test_gets_all_but_n_for_negative(self):
        self.assertEqual(u.get_n_jobs(-2), 3)

    def test_gets_all_for_zero(self):
        self.assertEqual(u.get_n_jobs(0), 4)
