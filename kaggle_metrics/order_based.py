import numpy as np
from sklearn.preprocessing import  binarize
from kaggle_metrics.utils import confusion_binary

# TODO:
# Area Under Curve (AUC)
# Gini
# Average Among Top P
# Average Precision (column-wise)
# Mean Average Precision (row-wise)
# [AveragePrecision@K] (row-wise)

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
auc = area_uder_curve
aatp = average_among_top_p
