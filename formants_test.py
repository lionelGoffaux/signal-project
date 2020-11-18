import unittest
import utils
import numpy as np


class FormantsTest(unittest.TestCase):
    def test_formants(self):
        x = np.arange(0, 20000)
        signal = np.sin(440/16000*2*np.pi*x)
        formants = utils.formants(signal, 21, 5, 16000)
        meanF1 = formants[:, 0].mean()
        meanF2 = formants[:, 1].mean()
        meanF3 = formants[:, 2].mean()
        meanF4 = formants[:, 3].mean()
        self.assertTrue(meanF1 <= meanF2)
        self.assertTrue(meanF2 <= meanF3)
        self.assertTrue(meanF3 <= meanF4)


if __name__ == "__main__":
    unittest.main()
