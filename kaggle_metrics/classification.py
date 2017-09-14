# Kaggle metrcis
# Krzysztof Joachimiak 2017

import numpy as np
import warnings
from utils import check_shapes, confusion_binary


# TODO: Classification metrics
# TODO: Check axes
# TODO: Check input

def log_loss(y_true, y_pred):
    '''

    Logarithmic loss

    Parameters
    ----------
    y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class probability

    Returns
    -------
    score: float
        Logarithmic loss score


    References
    ----------
    .. [1] https://www.kaggle.com/wiki/LogLoss
    .. [2] http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/
    .. [3] http://wiki.fast.ai/index.php/Log_Loss

    '''

    # Check shapes
    check_shapes(y_true, y_pred)

    return -(y_true - np.log(y_pred)).sum(axis=1).mean()

def mce(y_true, y_pred):
    '''

    Mean consequential error

    Parameters
    ----------
    y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class predictions (0 or 1 values only)

    Returns
    -------
    score: float
        Mean consequential error score

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/MeanConsequentialError
    .. [2] http://www.machinelearning.ru/wiki/images/5/59/PZAD2016_04_errors.pdf (RU)

    '''

    # TODO: check, if 0 or 1 values only
    # TODO: find papers

    # Check shapes
    check_shapes(y_true, y_pred)

    return (y_true.astype(bool) - y_pred.astype(bool)).mean()



def hamming_loss(y_true, y_pred):
    '''

    Hamming loss

    Parameters
     ----------
     y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class predictions (0 or 1 values only)

    Returns
    ------
    score: float
        Hamming loss score

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/HammingLoss
    .. [2] https://en.wikipedia.org/wiki/Multi-label_classification


    '''

    # TODO: check array shapes etc.

    # Check shapes
    check_shapes(y_true, y_pred)

    return np.logical_xor(y_pred, y_true).mean(axis=1).mean()


def mean_utility(y_true, y_pred, weights):

    '''

    Mean utility

    Parameters
     ----------
     y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class predictions (0 or 1 values only)

    Returns
    ------
    score: float
        Mean utility score

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/MeanUtility
    .. [2] https://en.wikipedia.org/wiki/Multi-label_classification

    Notes
    -----'
    The higher the better.

    '''

    # Check shapes
    check_shapes(y_true, y_pred)

    # Weights assignment
    w_tp, w_tn, w_fp, w_fn = weights

    # TODO: count these values
    tp, tn, fp, fn = confusion_binary(y_true, y_pred)

    return w_tp * tp + w_tn * tn + w_fp * fp + w_fn * fn


def mcc(y_true, y_pred):


    '''

    Matthews Correlation Coefficient

    Parameters
     ----------
     y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class predictions (0 or 1 values only)

    Returns
    ------
    score: float
        Matthews Correlation Coefficient score

    References
    ----------
    .. [1] https://lettier.github.io/posts/2016-08-05-matthews-correlation-coefficient.html
    .. [2] https://en.wikipedia.org/wiki/Matthews_correlation_coefficient



    '''




