# Kaggle metrcis
# Krzysztof Joachimiak 2017

import numpy as np
from sklearn.metrics import confusion_matrix

class WrongShapesExpception(Exception):
    pass


def check_shapes(y_true, y_pred):
    '''

    Check array shapes

    Parameters
    ----------
    y_true: numpy.ndarray
        Target array
    y_pred: numpy.ndarray
        Predicion array

    '''

    if not y_true.shape == y_pred.shape:
        raise WrongShapesExpception("Array shapes are inconsistent: "
                                    "y_true: {} "
                                    "y_pred: {}".format(y_true.shape, y_pred.shape))


def align_shape(*arrays):
    '''

    Make all the arrays 2-dimensional

    Parameters
    ----------
    arrays: list of numpy.ndarray
        Multiple arrays

    Returns
    -------
    reshaped_arrays: list of numpy.ndarray
        Arrays reshaped to (-1, 1)

    '''
    return [arr.reshape(-1, 1) for arr in arrays if arr.ndim == 1]

def is_binary(array):
    '''

    Check, if array contains only binary values

    Parameters
    ----------
    array: numpy.ndarray
        Numpy array

    Returns
    -------
    is_binary: bool
        Binary or not

    '''
    return np.array_equal(array, array.astype(bool))

def check_binary(y_true, y_pred):
    '''

    Check, if both prediction and target arrays are binary

    Parameters
    ----------
    y_true: numpy.ndarray
        Numpy array
    y_pred: numpy.ndarray
        Numpy array

    '''


    if not is_binary(y_pred):
        raise ValueError("Prediction array doesn't contain binary values only!")

    if not is_binary(y_true):
        raise ValueError("Ground truth array doesn't contain binary values only!")


def confusion_binary(y_true, y_pred):

    confmat = confusion_matrix(y_true, y_pred)

    confmat = confmat.astype(float)

    # TODO: check confmat
    true_negative = confmat[0, 0]
    false_negative = confmat[1, 0]

    true_positive = confmat[1, 1]
    false_positive = confmat[0, 1]

    return true_positive, true_negative, false_positive, false_negative

