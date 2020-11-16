import unittest
import utils
import numpy as np


class DistanceTest(unittest.TestCase):
    def test_get_distance(self):
        x = np.arange(5)
        y = np.array([0, 3, -4, 4, 0])
        self.assertEqual(2, utils.get_distance(x, y))
        y = np.zeros(5)
        self.assertEqual(-1, utils.get_distance(x, y))


if __name__ == "__main__":
    unittest.main()
