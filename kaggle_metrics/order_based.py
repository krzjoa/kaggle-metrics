
# TODO:
# Area Under Curve (AUC)
# Gini
# Average Among Top P
# Average Precision (column-wise)
# Mean Average Precision (row-wise)
# [AveragePrecision@K] (row-wise)

import numpy as np
from sklearn.preprocessing import  binarize
from kaggle_metrics.utils import confusion_binary


def auc(y_true, y_pred):
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





def aatp(y_true, y_pred):
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

    '''

def gini(y_tru, y_pred):
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

    '''
    pass

