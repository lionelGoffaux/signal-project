import unittest
import utils
import numpy as np


class MfccTest(unittest.TestCase):
    def test_mfcc(self):
        x = np.arange(0, 20000)
        signal = 300 * np.sin(500/16000*2*np.pi*x)
        mfcc = utils.mfcc(signal, 21, 5, 16000)
        self.assertTrue(abs(mfcc[:, 9].mean() - 38) < 1)


if __name__ == "__main__":
    unittest.main()
