import unittest
import utils
import numpy as np


class FuncDepTest(unittest.TestCase):

    def test_bidon(self):
        self.assertTrue(True)

    def test_energy(self):
        lst = np.ones(3)
        self.assertEqual(utils.energy(lst), 3)
        lst = np.arange(3)
        self.assertEqual(utils.energy(lst), 5)
        lst = np.array([1, -2, 3, -4])
        self.assertEqual(utils.energy(lst), 30)


if __name__ == "__main__":
    unittest.main()
