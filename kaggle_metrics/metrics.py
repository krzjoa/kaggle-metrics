# Kaggle metrcis
# Krzysztof Joachimiak 2017

import numpy as np
import warnings


# ============= REGRESSION METRICS ============= #

def mae(y_true, y_pred):

    '''

    Mean absolute error.

    Parameters
    ----------
    y_true: ndarray
        Ground truth
    y_pred: ndarray
        Array of predictions

    Returns
    -------
    rmsle: float
        Mean absolute error

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/MeanAbsoluteError

    '''

    return np.abs(y_true - y_pred).mean()


def wmae(y_true, y_pred, weights):

    '''

    Weighted mean absolute error.

    Parameters
    ----------
    y_true: ndarray
        Ground truth
    y_pred: ndarray
        Array of predictions

    Returns
    -------
    rmsle: float
        Weighted mean absolute error

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/WeightedMeanAbsoluteError

    '''

    return (weights * np.abs(y_true - y_pred)).mean()


def rmse(y_true, y_pred):
    '''

    Root mean squared error.

    Parameters
    ----------
    y_true: ndarray
        Ground truth
    y_pred: ndarray
        Array of predictions

    Returns
    -------
    rmsle: float
        Root mean squared error

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/RootMeanSquaredError

    '''


    return np.sqrt(((y_true - y_pred)**2).mean())


def rmsle(y_true, y_pred):
    '''

    Root mean squared logarithmic error.

    Parameters
    ----------
    y_true: ndarray
        Ground truth
    y_pred: ndarray
        Array of predictions

    Returns
    -------
    rmsle: float
        Root mean squared logarithmic error

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError

    '''

    _check_postive(y_pred)

    return np.sqrt((np.log(y_pred + 1) - np.log(y_true + 1)**2).mean())



# =============== CLASSIFICATION METRICS ================ #

# TODO: Classification metrics

def log_loss(y_true, y_pred):
    '''

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/LogarithmicLoss

    '''
    pass

def mce(y_true, y_pred):
    '''


    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/MeanConsequentialError

    '''
    pass


# class LogZeroException(Exception):
#     pass



def _check_postive(array):
    if (array < 0).sum():
        warnings.warn("Passed array contains at least one negative value. It may produce NaNs")