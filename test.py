import unittest
import numpy as np
import utils


class UtilsTest(unittest.TestCase):

    def test_split(self):
        sin = np.arange(80)

        with self.assertRaises(ValueError,
                               msg='negative width should raise a ValueError'):
            utils.split(sin, -250, 1000, 4)
        with self.assertRaises(ValueError,
                               msg='negative step should raise a ValueError'):
            utils.split(sin, 250, -1000, 4)

        with self.assertRaises(ValueError,
                               msg='negative width should raise a ValueError'):
            utils.split(sin, 0, 1000, 4)
        with self.assertRaises(ValueError,
                               msg='negative width should raise a ValueError'):
            utils.split(sin, 250, 0, 4)

        with self.assertRaises(ValueError,
                               msg='a width bellow the sampling period should'
                                   + ' raise a ValueError'):
            utils.split(sin, 249, 1000, 4)
        with self.assertRaises(ValueError,
                               msg='a step bellow the sampling period should'
                                   + ' raise a ValueError'):
            utils.split(sin, 249, 249, 4)

        with self.assertRaises(ValueError,
                               msg='a width greater than the signal should'
                                   + ' raise a ValueError'):
            utils.split(sin, 20250, 1000, 4)

        for n in range(4):
            res = utils.split(sin, (n+1)*250, 1000, 4)
            self.assertTupleEqual((20, n+1), res.shape,
                                  msg='the length of the frames'
                                      + ' are not corresponding')

        for n in range(4):
            res = utils.split(sin, 1000 + (n+1) * 250, 1000, 4)
            self.assertTupleEqual((19, 5+n), res.shape,
                                  msg='the overlap should change'
                                      + ' the number of frame')

        for n in range(4):
            res = utils.split(sin, 250, (n+1)*250, 4)
            self.assertTupleEqual((round(80/(n+1)), 1), res.shape,
                                  msg='the number of frames'
                                      + ' are not corresponding')

        for n in range(4):
            sin = np.arange(81 + n)
            res = utils.split(sin, 500, 1000, 4)
            if n == 0:
                self.assertTupleEqual((20, 2), res.shape,
                                      msg='if the rest is not is too small'
                                      + ' the rest should not be taken')
            else:
                self.assertTupleEqual((21, 2), res.shape,
                                      msg='if possible the rest'
                                      + ' should be taken')

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

    def test_get_distance(self):
        x = np.arange(5)
        y = np.array([0, 3, -4, 4, 0])
        self.assertEqual(2, utils.get_distance(x, y))
        y = np.zeros(5)
        self.assertEqual(-1, utils.get_distance(x, y))

    def test_autocorrelation(self):
        x = np.arange(0, 200)
        sig = np.sin(440/880*2*np.pi*x)
        corr = utils.autocorrelation(sig, 40, 5, 880, 5)
        for f in corr:
            self.assertEqual(f, 440.)


if __name__ == "__main__":
    unittest.main()
