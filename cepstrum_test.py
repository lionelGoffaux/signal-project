import unittest
import utils
import numpy as np


class CepstrumTest(unittest.TestCase):
    def test_cepstrum(self):
        x = np.arange(0, 20000)
        signal = np.sin(440/16000*2*np.pi*x)
        signal *= np.sin(880/16000*2*np.pi*x)
        signal *= np.sin(1320/16000*2*np.pi*x)
        signal *= np.sin(1760/16000*2*np.pi*x)
        corr = utils.get_pitch(signal, 21, 5, 16000, 5,
                               method=utils.cepstrum)
        for f in corr:
            self.assertTrue(abs(f - 440) < 5)


if __name__ == "__main__":
    unittest.main()
