import unittest
import numpy as np
import utils


class SplitTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
