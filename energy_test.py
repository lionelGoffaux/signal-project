import unittest
import numpy as np
import utils


class EnergyTest(unittest.TestCase):
    def test_energy(self):
        sin = np.zeros(3)
        self.assertEqual(0, utils.frame_energy(sin),
                         msg='The energy of null signal'
                         + ' should be 0')
        sin = np.ones(3)
        self.assertEqual(3, utils.frame_energy(sin))
        sin = np.arange(3)
        self.assertEqual(5, utils.frame_energy(sin))
        sin = np.array([1, -2, 3, -4])
        self.assertEqual(30, utils.frame_energy(sin))


if __name__ == "__main__":
    unittest.main()
