import unittest
import utils
import numpy as np


class AutocorrelationTest(unittest.TestCase):
    @unittest.skip("autocorrelation name refactor")
    def test_autocorrelation(self):
        x = np.arange(0, 200)
        sig = np.sin(440/880*2*np.pi*x)
        corr = utils.autocorrelation(sig, 40, 5, 880, 5)
        for f in corr:
            self.assertEqual(f, 440.)


if __name__ == "__main__":
    unittest.main()
