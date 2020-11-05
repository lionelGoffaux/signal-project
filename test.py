import unittest
import numpy as np
import utils


class FuncDepTest(unittest.TestCase):

    def test_split(self):
        sin = np.arange(8)
        frames = utils.split(sin, 500, 750, 4)
        print(frames.shape)

    def test_energy(self):
        lst = np.ones(3)
        self.assertEqual(utils.energy(lst), 3)
        lst = np.arange(3)
        self.assertEqual(utils.energy(lst), 5)
        lst = np.array([1, -2, 3, -4])
        self.assertEqual(utils.energy(lst), 30)


if __name__ == "__main__":
    unittest.main()
