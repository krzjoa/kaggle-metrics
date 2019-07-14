# TODO:
# Area Under Curve (AUC)
# Gini
# Average Among Top P
# Average Precision (column-wise)
# Mean Average Precision (row-wise)
# [AveragePrecision@K] (row-wise)

import numpy as np
from sklearn.preprocessing import binarize
from kaggle_metrics.utils import check_shapes, \
    confusion_binary, align_shape, check_binary


def average_precision_at_k(y_true, y_pred):
    # TODO: should work form matrix and vector as well
    '''

    Average precision

    Parameters
    ----------
    y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class predictions (0 or 1 values only)

    Returns
    ------
    score: numpy.ndarray
        Mean average precision score

    References
    ----------
    .. [1] https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    .. [2] https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

    '''
    true_positive = y_pred == y_true
    tp_cumsum = np.cumsum(true_positive)
    n_positive = y_true.sum()
    val_counter = np.cumsum(np.ones(len(y_pred)))
    return (tp_cumsum * true_positive / val_counter).sum() / n_positive



def mean_average_precision(y_true, y_pred):
    '''

    Mean average precision

    Parameters
     ----------
     y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class predictions (0 or 1 values only)

    Returns
    ------
    score: float
        Mean average precision score

    References
    ----------
    .. [1] https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    .. [2] https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision
    .. [3] https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

    '''

    # Check shapes
    check_shapes(y_true, y_pred)
    y_true, y_pred = align_shape(y_true, y_pred)


def area_uder_curve(y_true, y_pred):
    '''

    Area Under Curve (AUC)

    Parameters
    ----------
    y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class probability

    Returns
    -------
    auc_score: float
        AUC score

    References
    ----------
    .. [1] The Meaning and Use of the Area
            under a Receiver Operating
            Characteristic (ROC) Curve

            http://pubs.rsna.org/doi/pdf/10.1148/radiology.143.1.7063747
    '''

    for thr in np.arange(0.01, 1.01, 0.01):
        y_pred_bin = binarize(y_pred, thr)
        tp, tn, fp, fn = confusion_binary(y_true, y_pred)

def average_among_top_p(y_true, y_pred):
    '''

    Average Among Top P

    Parameters
    ----------
    y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class probability

    Returns
    -------
    aatp_score: float
        Average Among Top P score

    '''

def gini(y_tru, y_pred):
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

    '''
    pass

# Aliases
map = mean_average_precision
auc = area_uder_curve
aatp = average_among_top_p

if __name__ == "__main__":
    y_true = np.array([1,0, 1, 1, 1, 1, 1, 1, 0])
    y_pred = np.array([1,0, 1, 1, 0, 0, 1, 0, 0])

    print(average_precision(y_true, y_pred))