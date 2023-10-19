import unittest

import numpy as np

from scripts.value_profile_agreement import (compute_absolute_error,
                                             compute_cos, compute_kendall)


class TestComputeScores(unittest.TestCase):
    def setUp(self):
        self.profile_1 = np.array([1,0,0,0,0,0,0,0,0,0])
        self.profile_2 = np.array([1,0,0,0,0,0,0,0,0,0])
        self.profile_3 = np.array([0,0,0,0,0,0,0,0,0,10])
        self.profile_4 = np.array([0,10,10,10,10,10,10,10,10,10])

    def test_kendall(self):
        self.assertAlmostEqual(compute_kendall(self.profile_1, self.profile_2), 1.0)
        # kendall uses ordering so compute similarity so due to the nature of the profiles
        # we don't get the lowest similarity possible
        self.assertEqual(compute_kendall(self.profile_1, self.profile_3), 0.6)
        self.assertEqual(compute_kendall(self.profile_1, self.profile_4), 0.6)

    def test_cosine(self):
        self.assertAlmostEqual(compute_cos(self.profile_1, self.profile_2), 1.0)
        self.assertEqual(compute_cos(self.profile_1, self.profile_3), 0.0)
        self.assertEqual(compute_cos(self.profile_1, self.profile_4), 0.0)

    def test_absoluteerror(self):
        self.assertAlmostEqual(compute_absolute_error(self.profile_1, self.profile_2), 0.0)
        self.assertEqual(compute_absolute_error(self.profile_1, self.profile_3), 2.0)
        self.assertEqual(compute_absolute_error(self.profile_1, self.profile_4), 10.0)
