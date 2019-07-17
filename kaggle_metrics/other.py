# Kaggle metrcis
# Krzysztof Joachimiak 2017

import numpy as np


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