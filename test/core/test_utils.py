import os
import unittest
from unittest.mock import patch

import divik.core as u


@patch.object(os, os.cpu_count.__name__, new=lambda: 4)
class NJobsTest(unittest.TestCase):
    def test_gets_n_for_n_lower_than_cpus(self):
        assert u.get_n_jobs(3) == 3

    def test_gets_n_for_n_equal_cpus(self):
        assert u.get_n_jobs(4) == 4

    def test_gets_all_for_minus_one(self):
        assert u.get_n_jobs(-1) == 4

    def test_gets_all_but_n_for_negative(self):
        assert u.get_n_jobs(-2) == 3

    def test_gets_all_for_zero(self):
        assert u.get_n_jobs(0) == 4


class Dummy:
    def __init__(self, a, b=5):
        self.a = a
        self.b = b


class BuildTest(unittest.TestCase):
    def test_builds_instance_with_proper_args(self):
        dummy = u.build(Dummy, a=3, b=6)
        assert dummy.a == 3
        assert dummy.b == 6

        dummy = u.build(Dummy, a=3)
        assert dummy.a == 3
        assert dummy.b == 5

    def test_builds_with_too_many_args(self):
        dummy = u.build(Dummy, a=3, c=4)
        assert dummy.a == 3
        assert dummy.b == 5
