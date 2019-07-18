# Kaggle metrcis
# Krzysztof Joachimiak 2017

import numpy as np


def crps(y_true, y_pred):
    '''

    Continuous Ranked Probability Score

    Parameters
    ----------
    y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class predictions (0 or 1 values only)

    Returns
    ------
    score: float
        Continuous Ranked Probability Score score

    References
    ----------
    .. [1] https://www.kaggle.com/wiki/ContinuousRankedProbabilityScore
    .. [2] http://journals.ametsoc.org/doi/pdf/10.1175/1520-0434%282000%29015%3C0559%3ADOTCRP%3E2.0.CO%3B2

    '''

def intersection_over_union(y_true, y_pred):
    '''
    Intersection over union

    Parameters
    ----------
    y_true: numpy.ndarray
        Ground truth
    y_pred: numpy.ndarray
        Prediction

    Returns
    -------
    iou_score: float
        Intersection over union score

    '''
    intersection = y_true & y_pred
    union = y_true | y_pred
    return intersection.sum() / union.sum()

ioc = intersection_over_union
jaccard_index = intersection_over_union

if __name__ == '__main__':
    x = np.array([[0, 1, 1, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0]])

    y = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0]])

    print(ioc(x, y))