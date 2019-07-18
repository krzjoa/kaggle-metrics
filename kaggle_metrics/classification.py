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

    tn, fp, fn, tp = confusion_binary(y_true, y_pred)

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
    tn, fp, fn, tp = confusion_binary(y_true, y_pred)

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (fn + tn) * (fp + tn) * (tp + fn))

    return numerator / denominator

def roc_auc(y_true, y_pred, jump=0.01):
    '''
    Area under ROC (Receiver Operating Characteristics)  curve

    Parameters
    ----------
    y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class probability

    References
    ----------
    .. [1] https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

    Returns
    -------
    roc_auc_score: float
        ROC AUC score
    '''
    y_true, y_pred = y_true.reshape(-1, 1), y_pred.reshape(-1, 1)
    x = []
    y = []
    for thr in np.arange(0.01, 1 + jump, jump):
        y_pred_bin = binarize(y_pred, thr)
        tn, fp, fn, tp = confusion_binary(y_true, y_pred_bin)
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        y.append(tpr)
        x.append(fpr)
    x = np.array(x)
    y = np.array(y)
    return np.abs(np.trapz(y, x))


def gini(y_true, y_pred):
    '''
    Gini

    Parameters
    ----------
    y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class probability

    Returns
    -------
    gini_score: float
        Gini score

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
    .. [2] https://aichamp.wordpress.com/2017/10/19/calculating-auc-and-gini-model-metrics-for-logistic-classification/

    '''
    return 2 * roc_auc(y_true, y_pred) - 1

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
