# Kaggle metrcis
# Krzysztof Joachimiak 2017

import numpy as np
from kaggle_metrics.utils import check_shapes, \
    confusion_binary, align_shape, check_binary

# TODO: order of check_shapes and align_shapes


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
    y_true, y_pred = align_shape(y_true, y_pred)

    # Checking values
    if not (y_pred > 0).all():
        raise ValueError("Prediction array contains zeroes!")

    return -(y_true * np.log(y_pred)).sum(axis=1).mean()


def mean_consequential_error(y_true, y_pred):
    '''

    Mean consequential error
    Alias: mce

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

    Notes
    -----
    The higher the better.

    '''

    # Check shapes
    check_shapes(y_true, y_pred)
    y_true, y_pred = align_shape(y_true, y_pred)

    # Checking binarity
    check_binary(y_true, y_pred)

    return (y_true.astype(bool) == y_pred.astype(bool)).mean()


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

    Notes
    -----
    The smaller the better.

    '''

    # Check shapes
    check_shapes(y_true, y_pred)
    y_true, y_pred = align_shape(y_true, y_pred)

    # Logical values only!
    check_binary(y_true, y_pred)

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
    -----
    The higher the better.


    '''

    # Check shapes
    check_shapes(y_true, y_pred)
    y_true, y_pred = align_shape(y_true, y_pred)

    # Weights assignment
    w_tp, w_tn, w_fp, w_fn = weights

    tp, tn, fp, fn = confusion_binary(y_true, y_pred)

    return w_tp * tp + w_tn * tn + w_fp * fp + w_fn * fn


def matthews_correlation_coefficient(y_true, y_pred):
    '''

    Matthews Correlation Coefficient
    Alias: mcc

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

    # Check shapes
    check_shapes(y_true, y_pred)
    y_true, y_pred = align_shape(y_true, y_pred)

    # Confusion matrix values
    tp, tn, fp, fn = confusion_binary(y_true, y_pred)

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (fn + tn) * (fp + tn) * (tp + fn))

    return numerator / denominator


# Aliases
mce = mean_consequential_error
mcc = matthews_correlation_coefficient


if __name__ == '__main__':
    y_pred = np.array([1, 0.1, 0.2, 0.4, 0.23, 1, 0.34, 1, 1])
    y_true = np.array([1, 0, 0, 0, 1, 1, 0, 0, 1])

    print(log_loss(y_true, y_pred))

    # MCE loss
    y_pred = np.array([1, 0, 0, 0, 1, 1, 0, 1, 1])
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1])

    print(mce(y_true, y_pred))


    # Hamming loss
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1])
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1])

    print(hamming_loss(y_true, y_pred))


    # Average precision
