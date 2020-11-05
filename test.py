import unittest
import numpy as np
import utils


class FuncDepTest(unittest.TestCase):

    def test_split(self):
        sin = np.arange(8)
        frames = utils.split(sin, 500, 750, 4)
        print(frames.shape)


if __name__ == "__main__":
    unittest.main()
