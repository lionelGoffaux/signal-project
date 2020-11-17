import unittest
import utils
import numpy as np


class FormantsTest(unittest.TestCase):
    def test_formants(self):
        x = np.arange(0, 20000)
        signal = np.sin(440/16000*2*np.pi*x)
        formants = utils.formants(signal, 21, 5, 16000)
        self.assertTrue(abs(formants[:, 0].mean() - 440) < 2)


if __name__ == "__main__":
    unittest.main()
