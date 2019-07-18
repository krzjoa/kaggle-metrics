import numpy as np
from sklearn.preprocessing import binarize
from kaggle_metrics.utils import check_shapes, \
    confusion_binary, align_shape, check_binary


def average_precision_at_k(true_positive):
    # TODO: accept several types of input
    '''
    Average precision at position k

    Parameters
    ----------
    true_positive: numpy.ndarray
        True positive for ordered values in query

    Returns
    ------
    score: numpy.ndarray
        A vector of average precision score for every k-th point

    References
    ----------
    .. [1] https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    .. [2] https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

    '''
    tp_cumsum = np.cumsum(true_positive)
    val_counter = np.cumsum(np.ones(len(true_positive)))
    return np.cumsum(tp_cumsum * true_positive / val_counter) / tp_cumsum


def average_precision(true_positive):
    # TODO: find columnwise version of Average Precision
    '''
    Average precision

    Parameters
    ----------
    true_positive: numpy.ndarray
        True positive for ordered values in query

    Returns
    ------
    score: numpy.ndarray
        A vector of average precision score

    References
    ----------
    .. [1] https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    .. [2] https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

    '''
    return average_precision_at_k(true_positive)[-1]


def mean_average_precision(true_positive):
    '''
    Mean average precision

    Parameters
    ----------
    true_positive: numpy.ndarray
        True positive values for n queries (n_queries, answers)

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
    map_per_query = np.apply_along_axis(average_precision, 1, true_positive)
    return map_per_query.mean()



# Aliases
ap_at_k = average_precision_at_k
ap = average_precision
map = mean_average_precision

# TODO: try to find Average Among Top P (formerly described in one of Kaggle sites)

if __name__ == "__main__":
    y_pred = np.array([0.2, 0.3, 0.6, 0.7, 0.8, 0.1, 0.5, 0.55, 0.49])
    y_true = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0])
    y_true2 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_true3 = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 0, 1, 0, 0, 0],
                        [1, 1, 0, 0, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0, 1, 0, 0, 0]])
    # y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0])

    # print(average_precision_at_k(y_true2))
    # print(mean_average_precision(y_true3))

    print(roc_auc(y_true, y_pred))
    # from sklearn.metrics import roc_auc_score
    # print(roc_auc_score(y_true, y_pred))