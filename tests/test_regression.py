import unittest
import numpy as np
from kaggle_metrics.regression import *


class TestRegression(unittest.TestCase):

    # TODO: Additional tests

    def test_mse(self):
        y_pred = np.array([.1, 2., 3.4, 1., 5.3])
        y_true = np.array([.3, 2.2, 3.6, 1., 4.3])
        assert mae(y_true, y_pred) == 0.32

    def test_rmse(self):
        y_pred = np.array([.1, 2., 3.4, 1., 5.3])
        y_true = np.array([.3, 2.2, 3.6, 1., 4.3])
        assert rmse(y_true, y_pred) == 0.473286382648

    def test_rmsle(self):
        y_pred = np.array([.1, 2., 3.4, 1., 5.3])
        y_true = np.array([.3, 2.2, 3.6, 1., 4.3])
        assert rmsle(y_true, y_pred) == 0.113068903823



if __name__ == '__main__':
    unittest.main()
